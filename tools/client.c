/* SPDX-License-Identifier: Apache-2.0
 * bpf-gpu-server load client.
 *
 * Modes:
 *   own      one-at-a-time request/response against the bpf-gpu-server daemon
 *   burst    send N back-to-back, then collect N (pipeline depth)
 *   verify   send an all-ones vector and check the returned y against a CPU
 *            reference of the same 16x16x16 network
 *   paced    send at a fixed rate for a fixed duration (goodput)
 *
 * Timestamps: the client's own CLOCK_MONOTONIC value travels in the request
 * and is echoed in the response, so the round trip needs no server clock
 * trust. Server-internal phases (kernel parse -> GPU start, GPU start -> GPU
 * done) are reported from the response fields.
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "bpf_gpu_server_queue.h"

static uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

	return (x > y) - (x < y);
}

static void report(const char *name, uint64_t *v, int n)
{
	double s = 0;

	if (n <= 0) {
		printf("%-34s (no samples)\n", name);
		return;
	}
	qsort(v, n, sizeof(uint64_t), cmp_u64);
	for (int i = 0; i < n; i++)
		s += v[i];
	printf("%-34s n=%7d  p50=%8lu  p90=%8lu  p99=%8lu  max=%10lu  mean=%9.0f ns\n",
	       name, n, v[n / 2], v[n * 90 / 100], v[n * 99 / 100], v[n - 1],
	       s / n);
}

/* CPU reference of the GPU kernel's 16x16x16 leaky-ReLU network, same weights
 * (the kernel derives them from the index, so the reference does too). */
static void ref_forward(const float *x, float *y, int work, double *dbg)
{
	double w1[16][16], w2[16][16], w3[16][16];
	double a[16], b[16];

	for (int i = 0; i < 256; i++) {
		w1[i / 16][i % 16] = 0.01 * (double)((i * 7 + 1) % 13) - 0.05;
		w2[i / 16][i % 16] = 0.01 * (double)((i * 5 + 3) % 11) - 0.05;
		w3[i / 16][i % 16] = 0.01 * (double)((i * 3 + 5) % 17) - 0.08;
	}
	for (int i = 0; i < 16; i++)
		a[i] = x[i];
	for (int w = 0; w < work; w++) {
		for (int o = 0; o < 16; o++) {
			double s = 0;

			for (int i = 0; i < 16; i++)
				s += w1[o][i] * a[i];
			b[o] = s > 0 ? s : 0.01 * s;
		}
		for (int o = 0; o < 16; o++) {
			double s = 0;

			for (int i = 0; i < 16; i++)
				s += w2[o][i] * b[i];
			a[o] = s > 0 ? s : 0.01 * s;
		}
		for (int o = 0; o < 16; o++) {
			double s = 0;

			for (int i = 0; i < 16; i++)
				s += w3[o][i] * a[i];
			b[o] = s;
		}
		for (int i = 0; i < 16; i++)
			a[i] = b[i];
	}
	for (int i = 0; i < 16; i++)
		y[i] = (float)a[i];
	if (dbg)
		for (int i = 0; i < 16; i++)
			dbg[i] = a[i];
}

static void fill_x(float *x, uint32_t id)
{
	for (int i = 0; i < 16; i++)
		x[i] = 0.01f * (float)((int)(id * 13 + i * 7) % 37);
}

static int send_req(int s, struct sockaddr_in *to, uint32_t id)
{
	struct bgs_req_hdr r;

	memset(&r, 0, sizeof(r));
	r.magic = BGS_MAGIC;
	r.client_ns = now_ns();
	fill_x(r.x, id);
	return (int)sendto(s, &r, sizeof(r), 0, (struct sockaddr *)to,
			   sizeof(*to));
}

int main(int argc, char **argv)
{
	const char *mode = argc > 1 ? argv[1] : "own";
	int n = argc > 2 ? atoi(argv[2]) : 2000;
	int burst = argc > 3 ? atoi(argv[3]) : 8;
	int port = argc > 4 ? atoi(argv[4]) : 39400;
	int work = argc > 5 ? atoi(argv[5]) : 4;
	int run_ms = argc > 6 ? atoi(argv[6]) : 3000;
	struct sockaddr_in to;
	int s;
	uint64_t *rtt, *srv_pre, *srv_exec;
	struct bgs_resp_hdr resp;
	int got = 0, lost = 0, bad = 0;
	uint64_t loop_ms = 0;

	/* burst writes one rtt per response, so size for the largest run. */
	{
		size_t cap = (size_t)(n > 0 ? n : 200) *
				     (size_t)(burst > 0 ? burst : 1) +
			     16;

		rtt = (uint64_t *)malloc(sizeof(uint64_t) * cap);
		srv_pre = (uint64_t *)malloc(sizeof(uint64_t) * cap);
		srv_exec = (uint64_t *)malloc(sizeof(uint64_t) * cap);
	}
	(void)work;

	s = socket(AF_INET, SOCK_DGRAM, 0);
	memset(&to, 0, sizeof(to));
	to.sin_family = AF_INET;
	{
		/* Default loopback; BGS_DEST points the client at a veth/NIC
		 * address so a tc ingress hook on a real device is exercised. */
		const char *dst = getenv("BGS_DEST");

		to.sin_addr.s_addr = dst ? inet_addr(dst)
					 : htonl(INADDR_LOOPBACK);
	}
	to.sin_port = htons((uint16_t)port);
	{
		int sz = 8 << 20;
		struct timeval tv = {.tv_sec = 1, .tv_usec = 0};

		setsockopt(s, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
		/* Never block forever: a missing response must fail a run,
		 * not hang the bench script. */
		setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	}

	if (!strcmp(mode, "verify")) {
		int w = argc > 6 ? atoi(argv[6]) : 4;
		double ref[16], t0;
		uint32_t id = 424242;
		struct sockaddr_in from;
		socklen_t fl = sizeof(from);

		memset(&resp, 0, sizeof(resp));
		{
			struct bgs_req_hdr r;

			memset(&r, 0, sizeof(r));
			r.magic = BGS_MAGIC;
			r.client_ns = now_ns();
			/* A scale that keeps the 16x16 stack's output away from
			 * underflow, so the comparison is meaningful. */
			for (int i = 0; i < 16; i++)
				r.x[i] = 8.0f;
			(void)id;
			sendto(s, &r, sizeof(r), 0, (struct sockaddr *)&to,
			       sizeof(to));
			if (recvfrom(s, &resp, sizeof(resp), 0,
				     (struct sockaddr *)&from, &fl) <= 0) {
				perror("recvfrom");
				return 1;
			}
		}
		t0 = 0;
		(void)t0;
		{
			float x[16], y[16];

			for (int i = 0; i < 16; i++)
				x[i] = 8.0f;
			ref_forward(x, y, w, ref);
			printf("verify: GPU  y[0..3] = %.6f %.6f %.6f %.6f\n",
			       resp.y[0], resp.y[1], resp.y[2], resp.y[3]);
			printf("verify: CPU  y[0..3] = %.6f %.6f %.6f %.6f\n",
			       y[0], y[1], y[2], y[3]);
			double maxrel = 0;

			for (int i = 0; i < 16; i++) {
				double d = resp.y[i] - (double)y[i];
				double m = d < 0 ? -d : d;

				if (m > maxrel)
					maxrel = m;
			}
			printf("verify: max |GPU - CPU| = %.3e (%s)\n", maxrel,
			       maxrel < 1e-3 ? "OK" : "MISMATCH");
			printf("verify: resp magic=%08x id=%u client_ns_echo=%s\n",
			       resp.magic, resp.id,
			       resp.client_ns ? "yes" : "no");
		}
		return 0;
	}

	if (!strcmp(mode, "paced")) {
		uint64_t end = now_ns() + (uint64_t)run_ms * 1000000ull;
		uint64_t interval = n > 0 ? 1000000000ull / (uint64_t)n : 0;
		uint64_t next = now_ns();
		uint32_t id = 0;


		while (now_ns() < end) {
			while (now_ns() < next)
				;
			send_req(s, &to, id++);
			next += interval;
			/* Drain every iteration: a paced client that only
			 * reads 1 in 64 ends up with a full receive queue and
			 * measures its own reader, not the server. */
			for (;;) {
				int r = (int)recv(s, &resp, sizeof(resp),
						  MSG_DONTWAIT);

				if (r <= 0)
					break;
				got++;
			}
		}
		/* drain */
		while (recv(s, &resp, sizeof(resp), MSG_DONTWAIT) > 0)
			got++;
		printf("paced: sent=%u received=%d in %d ms -> %.0f req/s\n",
		       id, got, run_ms, (double)got * 1000.0 / run_ms);
		return 0;
	}

	if (!strcmp(mode, "burst")) {
		int d = burst; /* pipeline depth */
		int rounds = n > 0 ? n : 200;
		int sent = 0, recvd = 0, lost = 0, consec = 0;
		uint64_t t0 = now_ns(), t1 = 0;
		uint32_t id = 0;

		for (int r = 0; r < rounds && consec < 20; r++) {
			for (int i = 0; i < d; i++) {
				if (send_req(s, &to, id++) > 0)
					sent++;
			}
			for (int i = 0; i < d; i++) {
				struct sockaddr_in from;
				socklen_t fl = sizeof(from);
				int nr = (int)recvfrom(s, &resp, sizeof(resp), 0,
						       (struct sockaddr *)&from,
						       &fl);

				if (nr <= 0) {
					lost++;
					/* The server may exit mid-run; stop
					 * rather than burn a 1 s timeout per
					 * probe and hang the bench script. */
					if (++consec >= 20)
						break;
					continue;
				}
				consec = 0;
				recvd++;
				if (got < rounds * d)
					rtt[got++] = now_ns() - resp.client_ns;
			}
		}
		t1 = now_ns();
		printf("burst: depth=%d rounds=%d sent=%d received=%d lost=%d\n",
		       d, rounds, sent, recvd, lost);
		report("pipelined completion (client)", rtt, got);
		printf("throughput: %.0f req/s (depth %d)\n",
		       t1 > t0 ? (double)recvd * 1e9 / (double)(t1 - t0) : 0.0,
		       d);
		return 0;
	}

	/* own: one request in flight at a time (latency view) */
	{
		uint64_t loop0 = now_ns();
		int consec = 0;

		for (int i = 0; i < n; i++) {
			struct sockaddr_in from;
			socklen_t fl = sizeof(from);
			int r;

			if (send_req(s, &to, (uint32_t)i) <= 0) {
				lost++;
				continue;
			}
			r = (int)recvfrom(s, &resp, sizeof(resp), 0,
					  (struct sockaddr *)&from, &fl);
			if (r <= 0 || resp.magic != BGS_MAGIC ||
			    resp.y[0] != resp.y[0]) {
				lost++;
				/* The server may have exited mid-run; without
				 * this, n requests each burn the 1 s recv
				 * timeout and the bench script hangs. */
				if (++consec >= 20) {
					printf("client: aborting own after %d "
					       "consecutive misses\n", consec);
					break;
				}
				continue;
			}
			consec = 0;
			rtt[got] = now_ns() - resp.client_ns;
			if (resp.gpu_start_ns >= resp.ingress_ns)
				srv_pre[got] = resp.gpu_start_ns -
					       resp.ingress_ns;
			if (resp.gpu_done_ns >= resp.gpu_start_ns)
				srv_exec[got] = resp.gpu_done_ns -
						resp.gpu_start_ns;
			got++;
		}
		loop_ms = (now_ns() - loop0) / 1000000;
	}
	printf("client: mode=%s sent=%d got=%d lost=%d\n", mode, n, got, lost);
	report("round trip (client)", rtt, got);
	report("  parse -> GPU start", srv_pre, got);
	report("  GPU start -> GPU done", srv_exec, got);
	{
		double sum = 0;

		for (int i = 0; i < got; i++)
			sum += (double)rtt[i];
		printf("goodput: %.0f req/s (one at a time, 1/mean RTT)\n",
		       got && sum > 0 ? (double)got * 1e9 / sum : 0.0);
		printf("wall: %.0f req/s over %lu ms\n",
		       loop_ms ? (double)got * 1000.0 / (double)loop_ms : 0.0,
		       (unsigned long)loop_ms);
	}
	(void)bad;
	return 0;
}
