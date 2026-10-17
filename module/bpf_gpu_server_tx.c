// SPDX-License-Identifier: GPL-2.0-only
// Experimental kernel TCP TX for the shared token ring. The executor must
// publish DONE after its last page access; TX alone never reclaims PENDING.
#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/sched/cputime.h>
#include <linux/mutex.h>
#include <linux/delay.h>
#include <linux/bpf.h>
#include <linux/file.h>
#include <linux/net.h>
#include <linux/poll.h>
#include <net/inet_sock.h>
#include <linux/nsproxy.h>
#include <linux/uio.h>
#include <net/inet_hashtables.h>
#include <net/net_namespace.h>
#include <net/sock.h>
#include <net/tcp.h>
#include "../bpf/include/bpf_gpu_server_queue.h"
MODULE_IMPORT_NS("BPF_INTERNAL");

static int map_fd = -1;
module_param(map_fd, int, 0400);
MODULE_PARM_DESC(map_fd, "BPF map FD in the loading process (use bpf_gpu_server_load tx-load)");
static u32 reply_addr = 0x0100007fu;
module_param(reply_addr, uint, 0444);
MODULE_PARM_DESC(reply_addr, "fallback local IPv4 address without listener FD, network order");
static int listen_fd = -1;
module_param(listen_fd, int, 0400);
MODULE_PARM_DESC(listen_fd, "optional bootstrap listener FD for kernel accept/drain");
static struct socket *listener;
struct peer { struct list_head node; struct socket *socket; };
static LIST_HEAD(peers);
static unsigned long accepted, closed;
static struct net *bgs_net;
static struct task_struct *tx_task;
static DEFINE_MUTEX(cpu_lock);
static u64 stopped_cpu_ns;

/* Read-only scheduler accounting for this TX task, not total networking CPU. */
static int tx_cpu_get(char *buffer, const struct kernel_param *unused)
{
    u64 user = 0, system = 0;

    mutex_lock(&cpu_lock);
    if (tx_task && !IS_ERR(tx_task))
        task_cputime_adjusted(tx_task, &user, &system);
    else
        system = stopped_cpu_ns;
    mutex_unlock(&cpu_lock);
    return scnprintf(buffer, PAGE_SIZE, "%llu\n", user + system);
}
static const struct kernel_param_ops tx_cpu_ops = { .get = tx_cpu_get };
module_param_cb(tx_cpu_ns, &tx_cpu_ops, NULL, 0444);
MODULE_PARM_DESC(tx_cpu_ns, "read-only TX task adjusted user+system CPU nanoseconds");
static struct bpf_map *page_map;
static struct bgs_page *page;
static struct {
	struct sock *sk; /* lookup owns a reference, held until slot completion */
	u32 bytes;
	bool failed;
	u16 header_length, header_sent;
	char header[128];
} tx[BGS_LLM_SLOTS];
static unsigned long sent_bytes, completions, abandoned, retries;

static int reply_tcp(u32 idx, struct bgs_llm_slot *s, struct kvec *vec,
                     u32 nvec, u32 bytes, struct sock **locked)
{
	struct sock *sk = tx[idx].sk;
	struct msghdr msg = { .msg_flags = MSG_DONTWAIT | MSG_NOSIGNAL };
	int ret;

	if (!sk) {
		rcu_read_lock();
		sk = __inet_lookup_established(bgs_net, (__be32)s->addr_be,
			(__be16)s->port_be, (__be32)reply_addr, BGS_LLM_TCP_PORT, 0, 0);
		rcu_read_unlock();
		if (!sk)
			return -ENOENT;
		if (!sk_fullsock(sk)) {
			sock_gen_put(sk);
			return -ENOTCONN;
		}
		tx[idx].sk = sk;
	}
	iov_iter_kvec(&msg.msg_iter, ITER_SOURCE, vec, nvec, bytes);
	/* No RCU read section across lock_sock/send. No sk_socket dereference:
	 * close can detach that object even while a sock reference is held. */
	lock_sock(sk);
	*locked = sk;
	if (sock_flag(sk, SOCK_DEAD) || (sk->sk_shutdown & SEND_SHUTDOWN) ||
	    (sk->sk_state != TCP_ESTABLISHED && sk->sk_state != TCP_CLOSE_WAIT))
		ret = -ENOTCONN;
	else
		ret = tcp_sendmsg_locked(sk, &msg, bytes);
	/* Caller recycles a completed slot before releasing incoming backlog. */
	return ret;
}

/* Stream responses must retain request order on a persistent connection,
 * including ring wrap and a partially sent preceding response. */
static bool earlier_peer(u32 idx, struct bgs_llm_slot *s)
{
    for (u32 i = 0; i < BGS_LLM_SLOTS; i++) {
        struct bgs_llm_slot *other = &page->llm[i];
        u32 state;

        if (i == idx)
            continue;
        state = smp_load_acquire(&other->state);
        if ((state == BGS_PENDING || state == BGS_DONE) &&
            other->addr_be == s->addr_be && other->port_be == s->port_be &&
            other->ingress_ns < s->ingress_ns)
            return true;
    }
    return false;
}

/* Binary replies have no status header. Reject by ending the owned peer,
 * so the client observes EOF and the rest of the GPU queue remains live. */
static void reject_binary(struct bgs_llm_slot *s)
{
    struct peer *peer;

    list_for_each_entry(peer, &peers, node) {
        struct inet_sock *inet = inet_sk(peer->socket->sk);
        if ((__force u32)inet->inet_daddr == s->addr_be &&
            (__force u16)inet->inet_dport == s->port_be) {
            kernel_sock_shutdown(peer->socket, SHUT_RDWR);
            return;
        }
    }
}

static void poll_slot(u32 idx)
{
	struct bgs_llm_slot *s = &page->llm[idx];
	struct sock *locked = NULL, *put = NULL;
	u32 state = smp_load_acquire(&s->state);
	u32 produced;
	u16 flags = READ_ONCE(s->pad), transport = flags & BGS_LLM_TRANSPORT_MASK;
	bool header, rejected = flags & BGS_LLM_REJECTED;
	int ret;

	if ((state != BGS_PENDING && state != BGS_DONE) ||
	    (transport != 1 && transport != 2))
		return;
	/* A bodyless rejection header also completes a response. */
	if ((flags & BGS_LLM_VALIDATING) || (rejected && state != BGS_DONE))
		return;
	if (earlier_peer(idx, s))
		return;
	produced = smp_load_acquire(&s->produced);
	/* Keep the last token until GPU release. Receiving a whole response must
	 * not permit a new request while this slot is still executor-owned. */
	if (state != BGS_DONE && produced && produced == READ_ONCE(s->n_gen))
		produced--;
	if (produced > BGS_LLM_MAX_TOK || (!rejected && READ_ONCE(s->n_gen) > BGS_LLM_MAX_TOK))
		tx[idx].failed = true;
	if (rejected && transport == 1 && !tx[idx].failed) {
		reject_binary(s);
		tx[idx].failed = true;
	}
	if (transport == 2 && !tx[idx].header_length)
		tx[idx].header_length = scnprintf(tx[idx].header, sizeof(tx[idx].header),
			"HTTP/1.1 %s\r\nContent-Type: application/octet-stream\r\nContent-Length: %u\r\n\r\n",
			rejected ? "400 Bad Request" : "200 OK",
			rejected ? 0 : s->n_gen * (u32)sizeof(u32));
	header = tx[idx].header_sent < tx[idx].header_length;
	/* A valid HTTP header waits for the first available token so one send
	 * carries both. Bodyless rejection headers still complete immediately. */
	if (!tx[idx].failed && (!header || produced || rejected) &&
	    (header || produced * sizeof(u32) > tx[idx].bytes)) {
		struct kvec vec[2];
		u32 count = 0, header_bytes = 0, bytes = 0;
		if (header) {
			header_bytes = tx[idx].header_length - tx[idx].header_sent;
			vec[count++] = (struct kvec){ tx[idx].header + tx[idx].header_sent, header_bytes };
			bytes += header_bytes;
		}
		if (produced * sizeof(u32) > tx[idx].bytes) {
			u32 body_bytes = produced * sizeof(u32) - tx[idx].bytes;
			vec[count++] = (struct kvec){ (u8 *)s->tok_out + tx[idx].bytes, body_bytes };
			bytes += body_bytes;
		}
		ret = reply_tcp(idx, s, vec, count, bytes, &locked);
		if (ret > 0) {
			u32 sent_header = min_t(u32, ret, header_bytes);
			tx[idx].header_sent += sent_header;
			tx[idx].bytes += ret - sent_header; /* includes a partial u32 */
			sent_bytes += ret;
		} else {
			retries++;
			if (ret == -EPIPE || ret == -ECONNRESET || ret == -ENOTCONN ||
			    (ret == -ENOENT && state == BGS_DONE))
				tx[idx].failed = true;
		}
	}
	/* A disconnected peer cannot release memory still owned by an executor.
	 * DONE is its release; produced == n_gen is not a release. */
	if (state == BGS_DONE && (tx[idx].failed ||
	    (tx[idx].header_sent == tx[idx].header_length &&
	    tx[idx].bytes == produced * sizeof(u32)))) {
		put = tx[idx].sk;
		if (tx[idx].failed)
			abandoned++;
		else
			completions++;
		memset(&tx[idx], 0, sizeof(tx[idx]));
		smp_store_release(&s->state, BGS_FREE);
		/* Release the reservation last; a new owner may overwrite the slot
		 * immediately after this atomic operation. Preserve concurrent head
		 * advances and every other physical slot's reservation. */
		atomic64_fetch_andnot_release(1ULL << (32 + idx), (atomic64_t *)&page->llm_admission);
	}
	/* release_sock can run the next request's parser from TCP backlog. All
	 * old slot/TX accesses must finish first; keep the sock reference until
	 * after unlocking, even when completion cleared tx[idx].sk. */
	if (locked)
		release_sock(locked);
	if (put)
		sock_put(put);
}

/* The bootstrap process owns the listener file; sockfd_lookup pins it.
 * Accepted sockets and receive-buffer draining are entirely kernel-owned. */
static bool peer_pending(struct sock *sk)
{
	struct inet_sock *inet = inet_sk(sk);

	for (u32 i = 0; i < BGS_LLM_SLOTS; i++) {
		struct bgs_llm_slot *s = &page->llm[i];
		u32 state = smp_load_acquire(&s->state);

		if (state != BGS_FREE && ((s->pad & BGS_LLM_TRANSPORT_MASK) == 1 ||
                                  (s->pad & BGS_LLM_TRANSPORT_MASK) == 2) &&
		    s->addr_be == (__force u32)inet->inet_daddr &&
		    s->port_be == (__force u16)inet->inet_dport)
			return true;
	}
	return false;
}

static void accept_and_drain(void)
{
	struct socket *socket;
	struct peer *peer, *next;

	if (!listener)
		return;
	/* kernel_accept allocates even when the accept queue is empty. */
	while ((READ_ONCE(listener->ops)->poll(NULL, listener, NULL) & EPOLLIN) &&
	       kernel_accept(listener, &socket, O_NONBLOCK) == 0) {
		peer = kmalloc(sizeof(*peer), GFP_KERNEL);
		if (!peer) {
			sock_release(socket);
			continue;
		}
		peer->socket = socket;
		list_add_tail(&peer->node, &peers);
		accepted++;
	}
	list_for_each_entry_safe(peer, next, &peers, node) {
		char discard[256];
		struct kvec vec = { .iov_base = discard, .iov_len = sizeof(discard) };
		int ret;

		/* TCP poll includes sockmap ingress messages and EOF/error readiness. */
		if (!(READ_ONCE(peer->socket->ops)->poll(NULL, peer->socket, NULL) &
		      (EPOLLIN | EPOLLRDNORM | EPOLLRDHUP | EPOLLHUP | EPOLLERR)))
			continue;
		do {
			struct msghdr msg = {};

			ret = kernel_recvmsg(peer->socket, &msg, &vec, 1,
					     sizeof(discard), MSG_DONTWAIT);
		} while (ret > 0);
		if ((ret == 0 && !peer_pending(peer->socket->sk)) ||
		    (ret < 0 && ret != -EAGAIN && ret != -EWOULDBLOCK)) {
			list_del(&peer->node);
			sock_release(peer->socket);
			kfree(peer);
			closed++;
		}
	}
}

static int tx_thread(void *unused)
{
	while (!kthread_should_stop()) {
		accept_and_drain();
		for (u32 i = 0; i < BGS_LLM_SLOTS; i++)
			poll_slot(i);
		usleep_range(60, 120);
	}
	return 0;
}

static int __init bgs_init(void)
{
	struct bpf_array *array;
	int err;

	/* bpf_map_get validates the actual BPF FD and acquires a map reference.
	 * A bpffs filp_open FD is a seq_file, not a BPF map FD. */
	page_map = bpf_map_get(map_fd);
	if (IS_ERR(page_map))
		return PTR_ERR(page_map);
	if (page_map->map_type != BPF_MAP_TYPE_ARRAY ||
	    !(page_map->map_flags & BPF_F_MMAPABLE) ||
	    page_map->max_entries != 1 || page_map->value_size != sizeof(*page)) {
		err = -EINVAL;
		goto out_map;
	}
	array = container_of(page_map, struct bpf_array, map);
	page = (struct bgs_page *)array->value;
	bgs_net = get_net(current->nsproxy->net_ns);
	if (listen_fd >= 0) {
		listener = sockfd_lookup(listen_fd, &err);
		if (!listener)
			goto out_net;
		if (listener->type != SOCK_STREAM || listener->sk->sk_family != AF_INET ||
		    listener->sk->sk_protocol != IPPROTO_TCP ||
		    listener->sk->sk_state != TCP_LISTEN ||
		    inet_sk(listener->sk)->inet_num != BGS_LLM_TCP_PORT ||
		    !inet_sk(listener->sk)->inet_rcv_saddr ||
		    !net_eq(sock_net(listener->sk), bgs_net)) {
			err = -EINVAL;
			goto out_listener;
		}
		/* The bound listener is the authority for established-socket lookup. */
		reply_addr = (__force u32)inet_sk(listener->sk)->inet_rcv_saddr;
	}
	tx_task = kthread_run(tx_thread, NULL, "bpf_gpu_server_tx");
	if (IS_ERR(tx_task)) {
		err = PTR_ERR(tx_task);
		goto out_listener;
	}
	get_task_struct(tx_task);
	pr_info("bpf_gpu_server_tx: map id=%u size=%u ring=%zu slot=%zu net=%u\n",
		page_map->id, page_map->value_size, offsetof(struct bgs_page, llm),
		sizeof(struct bgs_llm_slot), bgs_net->ns.inum);
	return 0;
out_listener:
	if (listener)
		sockfd_put(listener);
out_net:
	put_net(bgs_net);
out_map:
	bpf_map_put(page_map);
	return err;
}

static void __exit bgs_exit(void)
{
	struct peer *peer, *next;

	u64 user, system;

	kthread_stop(tx_task);
	mutex_lock(&cpu_lock);
	task_cputime_adjusted(tx_task, &user, &system);
	stopped_cpu_ns = user + system;
	put_task_struct(tx_task);
	tx_task = NULL;
	mutex_unlock(&cpu_lock);
	list_for_each_entry_safe(peer, next, &peers, node) {
		list_del(&peer->node);
		sock_release(peer->socket);
		kfree(peer);
	}
	if (listener)
		sockfd_put(listener);
	for (u32 i = 0; i < BGS_LLM_SLOTS; i++)
		if (tx[i].sk)
			sock_put(tx[i].sk);
	bpf_map_put(page_map);
	put_net(bgs_net);
	pr_info("bpf_gpu_server_tx: exit bytes=%lu completed=%lu abandoned=%lu retries=%lu accepted=%lu closed=%lu\n",
		sent_bytes, completions, abandoned, retries, accepted, closed);
}
module_init(bgs_init);
module_exit(bgs_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("bpf-gpu-server experimental asynchronous kernel TCP reply");
