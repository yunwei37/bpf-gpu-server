// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server ingress-wake probe.
//
// Measures, on this host, the latency chain
//   sender sendto()  ->  eBPF socket filter  ->  userspace runnable
// for two wakeup mechanisms:
//   A) BPF_MAP_TYPE_RINGBUF consumed by a thread blocked in epoll_wait()
//      (the primitive a programmable eBPF request path can offer a GPU
//      executor today)
//   B) plain blocking recvfrom() on the socket (baseline)
//
// All timestamps are CLOCK_MONOTONIC (bpf_ktime_get_ns() is monotonic), so
// deltas are directly comparable on the same machine. The sender timestamp
// travels inside the packet payload and inside the ring-buffer record, so
// pairing is exact under loss/reordering.
#define _GNU_SOURCE
#include <sched.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

struct evt {
	uint64_t send_ts;
	uint64_t t_ns;
	uint32_t len;
	uint32_t pad;
};

#define CAP 200000
#define DEFAULT_PORT 39301

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ---------- measurement buckets ---------- */

struct series {
	uint64_t *v;
	int n;
	int cap;
};

static void push(struct series *s, uint64_t x)
{
	if (s->n < s->cap)
		s->v[s->n++] = x;
}

static int cmp_u64(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return (x > y) - (x < y);
}

static void report(const char *name, struct series *s)
{
	if (s->n == 0) {
		printf("%-32s (no samples)\n", name);
		return;
	}
	qsort(s->v, s->n, sizeof(uint64_t), cmp_u64);
	double sum = 0;
	for (int i = 0; i < s->n; i++)
		sum += s->v[i];
	printf("%-32s n=%6d  p50=%6lu  p90=%6lu  p99=%6lu  max=%8lu  mean=%7.0f ns\n",
	       name, s->n, s->v[s->n / 2], s->v[s->n * 90 / 100],
	       s->v[s->n * 99 / 100], s->v[s->n - 1], sum / s->n);
}

/* ---------- ring-buffer consumer ---------- */

struct rb_ctx {
	struct series to_bpf;   /* sendto -> BPF record */
	struct series to_user;  /* BPF record -> userspace callback */
	int total;
};

static int on_evt(void *ctx, void *data, size_t len)
{
	struct rb_ctx *c = ctx;
	struct evt *e = data;
	uint64_t t_user;

	if (len < sizeof(*e))
		return 0;
	t_user = now_ns(); /* as early as userspace sees the record */
	c->total++;
	if (getenv("PROBE_DEBUG") && c->total <= 3)
		fprintf(stderr, "evt send_ts=%lu t_ns=%lu len=%u load_ret=%d\n",
			(unsigned long)e->send_ts, (unsigned long)e->t_ns, e->len,
			(int)e->pad);
	if (e->send_ts && e->t_ns > e->send_ts)
		push(&c->to_bpf, e->t_ns - e->send_ts);
	if (t_user > e->t_ns)
		push(&c->to_user, t_user - e->t_ns);
	return 0;
}

/* ---------- sender ---------- */

struct send_arg {
	int fd;
	struct sockaddr_in dst;
	int n;
	int pace_us;
	int cpu; /* pin the sender here; -1 = no pin */
};

/* Optional isolation: pin the calling thread to a CPU and put it under
 * SCHED_FIFO. Finding 0001's p99 outliers were an *unpinned* upper bound;
 * this is the control experiment. */
static void pin_fifo(int cpu)
{
	cpu_set_t set;
	struct sched_param sp = {.sched_priority = 90};

	if (cpu < 0)
		return;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
		fprintf(stderr, "pin_fifo: sched_setscheduler: %s\n",
			strerror(errno));
}

static void *sender(void *arg)
{
	struct send_arg *a = arg;
	char buf[64];

	pin_fifo(a->cpu);

	for (int i = 0; i < a->n; i++) {
		uint64_t t = now_ns();
		memcpy(buf, &t, sizeof(t));
		sendto(a->fd, buf, sizeof(buf), 0, (struct sockaddr *)&a->dst,
		       sizeof(a->dst));
		if (a->pace_us > 0) {
			struct timespec d = {.tv_nsec = (long)a->pace_us * 1000};
			nanosleep(&d, NULL);
		}
	}
	return NULL;
}

static volatile int g_stop;
static int g_pin_cpu = -1;

/* Busy-wait mode: the consumer thread spins on the ring buffer instead of
 * blocking in epoll_wait(). Measures the same chain with the scheduler's
 * placement wakeup removed. */
static void *rb_spinner(void *arg)
{
	struct ring_buffer *rb = arg;

	pin_fifo(g_pin_cpu);
	while (!g_stop)
		ring_buffer__consume(rb);
	return NULL;
}


int main(int argc, char **argv)
{
	const char *obj = argc > 1 ? argv[1] : "bpf/wake_probe.bpf.o";
	int n_pkts = argc > 2 ? atoi(argv[2]) : 20000;
	int pace_us = argc > 3 ? atoi(argv[3]) : 50;
	int mode = argc > 4 ? atoi(argv[4]) : 1; /* 1=ringbuf, 0=recvfrom */
	int pin = argc > 5 ? atoi(argv[5]) : 0;  /* 1 = pin consumer+sender */
	int rx, tx;
	struct sockaddr_in addr;
	struct send_arg sa;
	pthread_t th;

	rx = socket(AF_INET, SOCK_DGRAM, 0);
	tx = socket(AF_INET, SOCK_DGRAM, 0);
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = htons(DEFAULT_PORT);
	if (bind(rx, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("bind");
		return 1;
	}
	sa.fd = tx;
	sa.dst = addr;
	sa.n = n_pkts;
	sa.pace_us = pace_us;
	sa.cpu = pin ? 3 : -1;
	/* In spin mode the consumer is the spinner thread; otherwise the main
	 * thread reads the ring. Pin whichever one actually consumes. */
	g_pin_cpu = (pin && mode == 2) ? 2 : -1;
	if (pin && mode != 2)
		pin_fifo(2);

	if (mode) {
		struct bpf_object *bo;
		struct bpf_program *prog;
		int prog_fd, rb_fd, ep, ret;
		struct ring_buffer *rb;
		struct rb_ctx ctx;
		struct epoll_event ev = {.events = EPOLLIN};
		int deadline_polls = 0;

		memset(&ctx, 0, sizeof(ctx));
		ctx.to_bpf.cap = ctx.to_user.cap = CAP;
		ctx.to_bpf.v = malloc(sizeof(uint64_t) * CAP);
		ctx.to_user.v = malloc(sizeof(uint64_t) * CAP);

		bo = bpf_object__open_file(obj, NULL);
		if (!bo) {
			fprintf(stderr, "open %s: %s\n", obj, strerror(errno));
			return 1;
		}
		ret = bpf_object__load(bo);
		if (ret) {
			fprintf(stderr, "load: %s\n", strerror(-ret));
			return 1;
		}
		prog = bpf_object__find_program_by_name(bo, "ingress");
		prog_fd = bpf_program__fd(prog);
		rb_fd = bpf_map__fd(bpf_object__find_map_by_name(bo, "events"));
		if (setsockopt(rx, SOL_SOCKET, SO_ATTACH_BPF, &prog_fd,
			       sizeof(prog_fd)) < 0) {
			perror("SO_ATTACH_BPF");
			return 1;
		}
		rb = ring_buffer__new(rb_fd, on_evt, &ctx, NULL);
		ep = epoll_create1(0);
		epoll_ctl(ep, EPOLL_CTL_ADD, rb_fd, &ev);

		pthread_create(&th, NULL, sender, &sa);
		if (mode == 2) {
			uint64_t deadline = now_ns() + 30000000000ull;
			pthread_t sp;

			pthread_create(&sp, NULL, rb_spinner, rb);
			/* The spinner owns the ring buffer; main just waits. */
			while (ctx.total < n_pkts && now_ns() < deadline) {
				struct timespec d = {.tv_nsec = 50000};
				nanosleep(&d, NULL);
			}
			g_stop = 1;
			pthread_join(sp, NULL);
		} else {
			while (ctx.total < n_pkts && deadline_polls < 200) {
				ret = epoll_wait(ep, &ev, 1, 100);
				if (ret <= 0) {
					deadline_polls++;
					continue;
				}
				ring_buffer__consume(rb);
			}
		}
		pthread_join(th, NULL);
		/* drain anything still queued */
		ring_buffer__consume(rb);

		printf("== %s: BPF ringbuf + %s (pace=%d us) ==\n",
		       mode == 2 ? "A2" : "A",
		       mode == 2 ? "busy-wait consume" : "userspace epoll_wait",
		       pace_us);
		printf("sent=%d  recorded=%d\n", n_pkts, ctx.total);
		report("sendto -> BPF record", &ctx.to_bpf);
		report("BPF record -> userspace wake", &ctx.to_user);
	} else {
		struct series rtt = {.v = malloc(sizeof(uint64_t) * CAP), .cap = CAP};
		char buf[2048];
		int seen = 0;

		pthread_create(&th, NULL, sender, &sa);
		while (seen < n_pkts) {
			uint64_t t0, t1, s;
			ssize_t r = recv(rx, buf, sizeof(buf), 0);
			t1 = now_ns();
			if (r < 8)
				break;
			memcpy(&s, buf, sizeof(s));
			t0 = s;
			if (t1 > t0)
				push(&rtt, t1 - t0);
			seen++;
		}
		pthread_join(th, NULL);
		printf("== B: baseline blocking recvfrom (pace=%d us) ==\n", pace_us);
		printf("sent=%d  received=%d\n", n_pkts, seen);
		report("sendto -> recvfrom return", &rtt);
	}
	return 0;
}
