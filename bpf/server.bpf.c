// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server: kernel-side ingress.
//
// A tc/clsact ingress program on the loopback device (the real ingress hook,
// same place a NIC's XDP-redirect path would land). For every well-formed
// request datagram it
//   1. copies the client's 64-byte float vector into the next control-page
//      slot, together with the client's clock, the parse time
//      (`bpf_ktime_get_ns()`) and the peer address parsed from the headers,
//   2. publishes the slot (payload, then state = PENDING, then head++),
//   3. pushes one byte into a ring buffer so a sleeping executor is woken
//      without polling.
//
// A slot is taken only if it is FREE *and* its completion has been consumed by
// the responder, which is what lets the GPU publish results into the same page
// without the producer clobbering them. Otherwise the datagram counts as a
// drop: there is no back-pressure channel from the GPU to a packet hook.
//
// The program returns TC_ACT_OK, so the datagram still reaches the UDP socket;
// the daemon drains it only to keep the receive queue from overflowing, and
// uses it as the ordinary fallback path for non-magic datagrams.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>
#include "bpf_gpu_server_admission.h"


#ifndef TC_ACT_OK
#define TC_ACT_OK 0
#endif
#define ETH_P_IP 0x0800
#define IPPROTO_UDP 17
#define IPPROTO_TCP 6

/* Doorbell: the record's presence is the wakeup, contents are unused. */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 12);
} doorbell SEC(".maps");

/* Diagnostics, read by the daemon at exit. */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 16);
	__type(key, __u32);
	__type(value, __u64);
} stats SEC(".maps");

static __always_inline void bump(__u32 key)
{
	__u64 *v = bpf_map_lookup_elem(&stats, &key);

	if (v)
		__sync_fetch_and_add(v, 1);
}

/* LLM request: header magic + n_prompt + n_gen + client_ns, then n_prompt
 * little-endian token ids. Written into the LLM ring; the executor generates. */
union bgs_llm_hdr {
	struct {
		__u32 magic;
		__u32 n_prompt;
		__u32 n_gen;
		__u32 pad;
		__u64 client_ns;
	};
	__u64 v;
};

/* Fill the next LLM slot from a request whose payload starts at `payload`.
 * `flags` = 0 for a UDP datagram (reply goes to addr_be/port_be), 1 for a TCP
 * segment (reply goes to the accepted socket that addr_be/port_be identify). */
static __always_inline int llm_fill(struct __sk_buff *skb, __u32 payload,
				     __u32 addr_be, __u16 port_be, __u16 flags)
{
	struct bgs_page *p;
	struct bgs_llm_slot *s;
	union bgs_llm_hdr h;
	__u32 tokens[BGS_LLM_MAX_TOK] = {}, zero = 0, i;

	if (bpf_skb_load_bytes(skb, payload, &h, sizeof(h)) != 0)
		return TC_ACT_OK;
	if (h.magic != BGS_LLM_MAGIC)
		return TC_ACT_OK;
	if (h.n_prompt == 0 || h.n_prompt > BGS_LLM_MAX_TOK)
		return TC_ACT_OK;
	if (h.n_gen == 0 || h.n_gen > BGS_LLM_MAX_TOK)
		return TC_ACT_OK;
	bump(7); /* llm magic */

	p = bpf_map_lookup_elem(&ctl, &zero);
	if (!p)
		return TC_ACT_OK;
	payload += sizeof(h);
#pragma unroll
	for (i = 0; i < BGS_LLM_MAX_TOK; i++) {
		if (i < h.n_prompt &&
		    bpf_skb_load_bytes(skb, payload + i * 4,
				       &tokens[i], 4) != 0)
			return TC_ACT_OK;
	}
	s = llm_reserve(p);
	if (!s) {
		__sync_fetch_and_add(&p->drops, 1);
		bump(8); /* llm busy */
		return TC_ACT_OK;
	}
	for (i = 0; i < BGS_LLM_MAX_TOK; i++) s->tok_in[i] = tokens[i];
	s->n_prompt = h.n_prompt;
	s->n_gen = h.n_gen;
	s->produced = 0;
	s->pad = flags;
	s->addr_be = addr_be;
	s->port_be = port_be;
	s->client_ns = h.client_ns;
	s->ingress_ns = bpf_ktime_get_ns();
	s->state = BGS_PENDING;
	return TC_ACT_OK;
}

static __always_inline int llm_ingress(struct __sk_buff *skb, __u32 l4,
				       struct udphdr *udp, struct iphdr *ip)
{
	return llm_fill(skb, l4 + sizeof(*udp), ip->saddr, udp->source, 0);
}

static __always_inline int llm_ingress_tcp(struct __sk_buff *skb, __u32 payload,
					   struct tcphdr *tcp, struct iphdr *ip)
{
	return llm_fill(skb, payload, ip->saddr, tcp->source, 1);
}

SEC("classifier")
int ingress(struct __sk_buff *skb)
{
	struct bgs_page *p;
	struct bgs_req_hdr h;
	struct iphdr ip;
	struct udphdr udp;
	__u32 head, idx, zero = 0, l4;
	struct bgs_ctl_slot *s;
	void *d;

	bump(0); /* seen */
	if (skb->protocol != bpf_htons(ETH_P_IP))
		return TC_ACT_OK;
	/* Loopback uses the usual 14-byte Ethernet header on this kernel; a
	 * raw-IP path (no L2) is handled by the fallback below. */
	if (bpf_skb_load_bytes(skb, 12, &udp, 2) == 0 &&
	    *(__be16 *)&udp != bpf_htons(ETH_P_IP))
		return TC_ACT_OK;
	if (bpf_skb_load_bytes(skb, 14, &ip, sizeof(ip)) != 0)
		return TC_ACT_OK;
	if (ip.protocol != IPPROTO_UDP) {
		struct tcphdr tcp;
		__u32 thl;

		if (ip.protocol != IPPROTO_TCP)
			return TC_ACT_OK;
		bump(1);
		l4 = 14 + (__u32)(ip.ihl * 4);
		if (bpf_skb_load_bytes(skb, l4, &tcp, sizeof(tcp)) != 0)
			return TC_ACT_OK;
		/* Only the first payload segment matters: the client sends the
		 * whole LLM request in one write. */
		if (tcp.dest != bpf_htons(BGS_LLM_TCP_PORT))
			return TC_ACT_OK;
		bump(9); /* tcp dest match */
		thl = (__u32)(tcp.doff * 4);
		if (bpf_skb_load_bytes(skb, l4 + thl, &h,
				       sizeof(h.magic)) != 0)
			return TC_ACT_OK;
		if (h.magic != BGS_LLM_MAGIC)
			return TC_ACT_OK;
		bump(10); /* tcp magic */
		return llm_ingress_tcp(skb, l4 + thl, &tcp, &ip);
	}
	l4 = 14 + (__u32)(ip.ihl * 4);
	if (bpf_skb_load_bytes(skb, l4, &udp, sizeof(udp)) != 0)
		return TC_ACT_OK;
	if (udp.dest == bpf_htons(BGS_LLM_PORT))
		return llm_ingress(skb, l4, &udp, &ip);
	if (udp.dest != bpf_htons(BGS_PORT))
		return TC_ACT_OK;
	bump(1); /* ip+udp */

	if (bpf_skb_load_bytes(skb, l4 + sizeof(udp), &h, sizeof(h)) != 0)
		return TC_ACT_OK;
	if (h.magic != BGS_MAGIC)
		return TC_ACT_OK;
	bump(2); /* magic */

	p = bpf_map_lookup_elem(&ctl, &zero);
	if (!p)
		return TC_ACT_OK;

	head = p->head;
	idx = head % BGS_SLOTS;
	s = &p->slots[idx];
	if (s->state != BGS_FREE || p->done[idx].state != BGS_FREE) {
		p->drops++;
		bump(3);
		bump(s->state != BGS_FREE ? 5 : 6); /* 5: ctl busy, 6: done busy */
		goto out;
	}
	/* bgs_req_hdr: magic(4) pad(4) client_ns(8), then x[] at offset 16. */
	if (bpf_skb_load_bytes(skb, l4 + sizeof(udp) + 16, s->x,
			       sizeof(s->x)) != 0)
		goto out;
	s->id = head;
	s->client_ns = h.client_ns;
	s->ingress_ns = bpf_ktime_get_ns();
	p->peers[idx].addr_be = ip.saddr;
	p->peers[idx].port_be = udp.source;
	s->state = BGS_PENDING;
	__sync_fetch_and_add(&p->head, 1);
	bump(4); /* published */
out:
	d = bpf_ringbuf_reserve(&doorbell, 1, 0);
	if (d)
		bpf_ringbuf_submit(d, 0);
	return TC_ACT_OK;
}


/* TCP sequencing/reassembly precedes this stream parser. */
struct bgs_flow { __u32 local, remote, local_port, remote_port; };
struct {
    __uint(type, BPF_MAP_TYPE_SOCKHASH);
    __uint(max_entries, BGS_SLOTS);
    __type(key, struct bgs_flow);
    __type(value, __u32);
} streams SEC(".maps");

SEC("sockops")
int stream_sockops(struct bpf_sock_ops *ctx)
{
    if (ctx->op == BPF_SOCK_OPS_PASSIVE_ESTABLISHED_CB &&
        ctx->family == 2 && ctx->local_port == BGS_LLM_TCP_PORT) {
        struct bgs_flow key = { ctx->local_ip4, ctx->remote_ip4,
                              ctx->local_port, ctx->remote_port };
        int error=bpf_sock_hash_update(ctx, &streams, &key, BPF_NOEXIST);
        if (error) {
            __u32 zero=0;
            struct bgs_page *page=bpf_map_lookup_elem(&ctl,&zero);
            if(page)__sync_val_compare_and_swap(&page->admission_pad,0,(__u32)-error);
            bump(13);
        }
        else bump(12);
    }
    return 0;
}

extern void *bpf_cast_to_kern_ctx(void *ctx) __ksym;

/* strparser can clone one skb for several messages. load_bytes offsets are
 * skb-relative; the message does not necessarily start at its data[0]. */
static __always_inline __u32 stream_offset(struct __sk_buff *ctx)
{
    struct sk_buff *skb=bpf_cast_to_kern_ctx(ctx);
    struct sk_skb_cb *cb=(void *)skb->cb;
    return BPF_CORE_READ(cb,strp.strp.offset);
}

/* Minimal Content-Length POST transport for the existing binary token API.
 * bpf_loop follows available stream bytes; no separate header-size tuning. */
struct http_scan {
    __u32 rolling, line_pos, end, length, value;
    int match, reading, digits, have_length, error;
};
__noinline int http_step(struct http_scan *s, __u8 c)
{
    if (!s) return 1;
    const char name[]="content-length";
    __u8 lower=c>='A' && c<='Z' ? c+32 : c;
    if (c=='\n') {
        if (s->reading) {
            if (!s->digits || s->have_length) { s->error=1; return 1; }
            s->length=s->value; s->have_length=1;
        }
        s->line_pos=0; s->match=0; s->reading=0; s->digits=0; s->value=0;
    } else {
        if (s->line_pos<14) {
            if (s->match>=0 && lower==name[s->line_pos]) s->match++;
            else s->match=-1;
        } else if (s->line_pos==14 && c==':' && s->match==14) {
            s->reading=1;
        } else if (s->reading) {
            if (c>='0' && c<='9') {
                if (s->value>sizeof(union bgs_llm_hdr)+BGS_LLM_MAX_TOK*4) { s->error=1; return 1; }
                s->value=s->value*10+c-'0'; s->digits=1;
            } else if (c!=' ' && c!='\t' && c!='\r') { s->error=1; return 1; }
        }
        if (s->line_pos<15) s->line_pos++; /* only header-name positions matter */
    }
    s->rolling=(s->rolling<<8)|c;
    if (s->rolling==0x0d0a0d0a) { return 1; }
    return 0;
}
struct http_scan_ctx {
    struct __sk_buff *skb;
    __u32 base;
    struct http_scan state;
};
static long http_byte(__u32 off, void *opaque)
{
    struct http_scan_ctx *s=opaque;
    __u8 c;
    if (bpf_skb_load_bytes(s->skb,s->base+off,&c,1)) return 1;
    int done=http_step(&s->state,c);
    if (done && !s->state.error) s->state.end=off+1;
    /* Global-function verification does not propagate return bounds. */
    asm volatile("%0 &= 1" : "+r"(done));
    return done;
}
static __always_inline int frame_info(struct __sk_buff *skb, __u32 *body, __u16 *http)
{
    __u32 magic, offset=stream_offset(skb);
    union bgs_llm_hdr h;
    *body=0; *http=0;
    if (offset>skb->len) return -1;
    __u32 available=skb->len-offset;
    if (available<4 || bpf_skb_load_bytes(skb,offset,&magic,4)) return 0;
    if (magic==0x54534f50) { /* POST */
        struct http_scan_ctx ctx={.skb=skb,.base=offset};
        struct http_scan *scan=&ctx.state;
        if (bpf_loop(available,http_byte,&ctx,0)<0 || scan->error) return -1;
        if (!scan->end) return 0;
        if (!scan->have_length || scan->length<sizeof(h) ||
            scan->length>sizeof(h)+BGS_LLM_MAX_TOK*4) return -1;
        *body=scan->end; *http=1;
        return scan->end+scan->length;
    }
    if (available<sizeof(h) || bpf_skb_load_bytes(skb,offset,&h,sizeof(h))) return 0;
    if (h.magic!=BGS_LLM_MAGIC || !h.n_prompt || h.n_prompt>BGS_LLM_MAX_TOK ||
        !h.n_gen || h.n_gen>BGS_LLM_MAX_TOK) return -1;
    return sizeof(h)+h.n_prompt*4;
}
SEC("sk_skb/stream_parser")
int stream_parse(struct __sk_buff *skb)
{
    __u32 body; __u16 http;
    int length;
    bump(14);
    length=frame_info(skb,&body,&http);
    if (length<0) bump(15);
    return length;
}

struct {
    __uint(type,BPF_MAP_TYPE_ARRAY);
    __uint(max_entries,1);
    __type(key,__u32);
    __type(value,struct bgs_uring_context);
} uring_owner SEC(".maps");

static __always_inline int stream_dispatch(struct __sk_buff *skb, int queued)
{
    union bgs_llm_hdr h;
    struct bgs_llm_pending request={};
    __u32 zero=0, body, offset=stream_offset(skb);
    __u16 http;
    int length=frame_info(skb,&body,&http);
    if (length<=0 || offset>skb->len || length>skb->len-offset ||
        bpf_skb_load_bytes(skb,offset+body,&h,sizeof(h)) ||
        h.magic!=BGS_LLM_MAGIC || length!=body+sizeof(h)+h.n_prompt*4 ||
        !h.n_prompt || h.n_prompt>BGS_LLM_MAX_TOK ||
        !h.n_gen || h.n_gen>BGS_LLM_MAX_TOK) return SK_DROP;
#pragma unroll
    for (__u32 i=0;i<BGS_LLM_MAX_TOK;i++)
        if (i<h.n_prompt && bpf_skb_load_bytes(skb,offset+body+sizeof(h)+i*4,&request.tokens[i],4)) return SK_DROP;
    struct bgs_page *p=bpf_map_lookup_elem(&ctl,&zero);
    if (!p || *(volatile __u64 *)&p->stop_ns) return SK_DROP;
    request.n_prompt=h.n_prompt; request.n_gen=h.n_gen;
    request.flags=(http ? 2 : 1) | BGS_LLM_VALIDATING;
    request.addr_be=skb->remote_ip4; request.port_be=(__u16)(skb->remote_port>>16);
    request.client_ns=h.client_ns; request.ingress_ns=bpf_ktime_get_ns();
    if (queued) {
        /* Count before publishing so EOF cannot recycle a connection while
         * one of its requests is waiting outside the model's eight slots. */
        __u32 bucket=bgs_pending_bucket(request.addr_be,request.port_be);
        __sync_fetch_and_add(&p->llm_peer_pending[bucket],1);
        __sync_fetch_and_add(&p->llm_queued,1);
        if (bpf_map_push_elem(&pending,&request,BPF_ANY)) {
            __sync_fetch_and_sub(&p->llm_queued,1);
            __sync_fetch_and_sub(&p->llm_peer_pending[bucket],1);
            bump(8); __sync_fetch_and_add(&p->drops,1); return SK_DROP;
        }
    } else {
        struct bgs_llm_slot *slot=llm_reserve(p);
        if (!slot) { bump(8); __sync_fetch_and_add(&p->drops,1); return SK_DROP; }
        llm_publish(slot,&request);
    }
    bump(11);
    /* Preserve qualified sockmap/TCP receive-cursor accounting. */
    struct bgs_flow key={skb->local_ip4,skb->remote_ip4,skb->local_port,skb->remote_port};
    return bpf_sk_redirect_hash(skb,&streams,&key,BPF_F_INGRESS);
}
SEC("sk_skb/stream_verdict")
int stream_publish(struct __sk_buff *skb)
{
    return stream_dispatch(skb,0);
}
SEC("sk_skb/stream_verdict")
int stream_uring_publish(struct __sk_buff *skb)
{
    __u32 zero=0;
    struct bgs_uring_context *owner=bpf_map_lookup_elem(&uring_owner,&zero);
    return stream_dispatch(skb,owner && owner->pad);
}

char _license[] SEC("license") = "GPL";

SEC("tracepoint/io_uring/io_uring_create")
int uring_bind(struct trace_event_raw_io_uring_create *ctx)
{
    __u32 zero=0;
    struct bgs_uring_context *owner=bpf_map_lookup_elem(&uring_owner,&zero);
    struct bpf_pidns_info ns={};
    if(owner && !bpf_get_ns_current_pid_tgid(owner->nsdev,owner->nsino,&ns,sizeof(ns)) && owner->pid==ns.tgid)
        __sync_val_compare_and_swap(&owner->ring,0,(__u64)ctx->ctx);
    return 0;
}
SEC("tracepoint/io_uring/io_uring_complete")
int uring_recycle(struct trace_event_raw_io_uring_complete *ctx)
{
    __u32 zero=0;
    struct bgs_uring_context *owner=bpf_map_lookup_elem(&uring_owner,&zero);
    if(!owner || !owner->ring || owner->ring!=(__u64)ctx->ctx)return 0;
    __u64 tag=ctx->user_data;__u32 prefix=tag>>48;
    if(prefix!=BGS_IO_SEND_TAG && prefix!=BGS_IO_ABORT_TAG)return 0;
    __u32 length=(tag>>32)&0xffff,sequence=(__u32)tag;
    if((prefix==BGS_IO_SEND_TAG && (!length || ctx->res!=length)) ||
       (prefix==BGS_IO_ABORT_TAG && ctx->res!=0))return 0;
    struct bgs_page *page=bpf_map_lookup_elem(&ctl,&zero);
    if(!page)return 0;
    __u32 index=sequence%BGS_LLM_SLOTS;
    struct bgs_llm_slot *slot=&page->llm[index];
    if(slot->state!=BGS_DONE)return 0;
    slot->state=BGS_FREE;
    __sync_fetch_and_and(&page->llm_admission,~(1ULL<<(32+index)));
    bump(4);
    return 0;
}
