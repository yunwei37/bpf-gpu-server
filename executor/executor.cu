// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server: resident daemon = GPU executor + packet gateway.
//
// One process, two threads, one shared page:
//
//   main thread (GPU, pinned)     consumes PENDING slots from the mmap'able
//                                 control page written by the socket filter,
//                                 runs one kernel per batch, publishes the
//                                 completions back into the same page and
//                                 bumps the futex word (done_seq).
//   responder thread (pinned)     sleeps on done_seq with FUTEX_WAIT, and for
//                                 every published completion sends the
//                                 response datagram to the peer address the
//                                 kernel recorded. It also drains the ingress
//                                 socket, so datagrams do not pile up in the
//                                 kernel after the filter returns -1.
//
// There is no control plane: no per-request syscall on the ingress side, no
// thread per connection, no lock. The page is the queue.
#include <arpa/inet.h>
#include <errno.h>
#include <linux/futex.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

#include "cuda_timer.h"
#include "bpf_gpu_server_queue.h"

// One request: `work` passes of a 16x16x16 triple with a leaky-ReLU, so the
// measured latency is arithmetic rather than pure launch overhead.
//
// The kernel operates **directly on the BPF control page**: that page is
// registered as pinned, mapped host memory (cudaHostRegisterMapped) and the
// device pointer aliases the same physical pages the eBPF ingress program
// wrote. There is no per-batch staging copy. `base` is the global request
// index of the first slot in this batch, and `offset` converts the GPU
// %globaltimer into CLOCK_MONOTONIC on the GPU side so the host never has to
// rewrite the timestamps.
__global__ void bgs_exec_kernel(struct bgs_page *p, unsigned int work,
			       long long offset)
{
	__shared__ float w1[16][16], w2[16][16], w3[16][16];
	int t = threadIdx.x;

	for (int i = t; i < 256; i += blockDim.x) {
		w1[i / 16][i % 16] = 0.01f * (float)((i * 7 + 1) % 13) - 0.05f;
		w2[i / 16][i % 16] = 0.01f * (float)((i * 5 + 3) % 11) - 0.05f;
		w3[i / 16][i % 16] = 0.01f * (float)((i * 3 + 5) % 17) - 0.08f;
	}
	__syncthreads();
	if (t >= 16)
		return;

	/* Resident: this kernel is launched **once** for the whole run and
	 * spins on the mmap'ed page it reads/writes directly. The host never
	 * launches per request; the only steady-state control-plane action
	 * left is the ingress doorbell (a future `bpf_send_signal`/kernel-side
	 * notify). Spin costs GPU issue slots (<=1 SM) and no host CPU. */
	for (;;) {
		__u32 idx;
		struct bgs_done_slot *out;
		struct bgs_ctl_slot *in;
		unsigned long long t_start, t_done;
		float a[16], b[16];
		/* Diagnostic: count spins in the unused pad0 word so a host
		 * can prove the resident kernel is live. */
		__stcg((unsigned int *)&p->pad0,
		       __ldcg((const unsigned int *)&p->pad0) + 1);

		/* Volatile loads still hit the SM's L1, which is not
		 * coherent with host memory, so the resident kernel would
		 * spin on stale zeros forever. __ldcg reads at gpu scope
		 * (bypassing L1); the matching writes use __stcg. */
		if (__ldcg((const unsigned long long *)&p->stop_ns) &&
		    (unsigned long long)((long long)bgs_globaltimer() + offset) >=
			    __ldcg((const unsigned long long *)&p->stop_ns))
			return;
		if (__ldcg((const unsigned int *)&p->done_seq) >=
		    __ldcg((const unsigned int *)&p->head))
			continue;
		idx = __ldcg((const unsigned int *)&p->done_seq) % BGS_SLOTS;
		in = &p->slots[idx];
		out = &p->done[idx];
		/* `BGS_PENDING` in the ctl slot means the BPF producer has
		 * published; `BGS_FREE` in the done slot means the responder
		 * has finished with a previous round and recycled it, so the
		 * GPU may overwrite it without racing the reader. */
		if (__ldcg((const unsigned int *)&in->state) != BGS_PENDING ||
		    __ldcg((const unsigned int *)&out->state) != BGS_FREE)
			continue;
		if (t == 0)
			t_start = bgs_globaltimer();
		for (int i = 0; i < BGS_VEC; i++)
			a[i] = __ldcg((const float *)&in->x[i]);
		for (unsigned int w = 0; w < work; w++) {
			for (int o = 0; o < 16; o++) {
				float s = 0;

				for (int i = 0; i < 16; i++)
					s += w1[o][i] * a[i];
				b[o] = s > 0 ? s : 0.01f * s;
			}
			for (int o = 0; o < 16; o++) {
				float s = 0;

				for (int i = 0; i < 16; i++)
					s += w2[o][i] * b[i];
				a[o] = s > 0 ? s : 0.01f * s;
			}
			for (int o = 0; o < 16; o++) {
				float s = 0;

				for (int i = 0; i < 16; i++)
					s += w3[o][i] * a[i];
				b[o] = s;
			}
			for (int i = 0; i < 16; i++)
				a[i] = b[i];
		}
		if (t == 0) {
			t_done = bgs_globaltimer();
			out->id = __ldcg((const unsigned int *)&in->id);
			out->client_ns =
				__ldcg((const unsigned long long *)&in->client_ns);
			out->ingress_ns =
				__ldcg((const unsigned long long *)&in->ingress_ns);
			for (int i = 0; i < BGS_VEC; i++)
				out->y[i] = a[i];
			out->gpu_start_ns =
				(unsigned long long)((long long)t_start + offset);
			out->gpu_done_ns =
				(unsigned long long)((long long)t_done + offset);
			__threadfence_system();
			__stcg((unsigned int *)&out->state, BGS_DONE);
			__stcg((unsigned int *)&in->state, BGS_FREE);
			__stcg((unsigned int *)&p->served,
			       __ldcg((const unsigned int *)&p->served) + 1);
			__stcg((unsigned int *)&p->done_seq,
			       __ldcg((const unsigned int *)&p->done_seq) + 1);
		}
	}
}

struct daemon {
	struct bgs_page *page;
	int sock;
	long pg;
	int run_seconds;
	int work;
	int batch;
	int cpu_gpu;
	int cpu_responder;
	int idle_mode; /* 0 futex, 1 spin, 2 stop after settle */

	unsigned long n_busy;  /* GPU passes that had a batch ready */
	unsigned long n_idle;  /* passes that had to wait for the doorbell */
	/* Honest bubble accounting:
	 *   busy_ns - host wall time the GPU loop was staging or executing a
	 *             batch (i.e. at least one request was queued);
	 *   gpu_ns  - kernel execution time summed from the GPU clock;
	 *   wait_ns - time the loop spent waiting for the doorbell.
	 * The exposed bubble is busy_ns - gpu_ns. */
	uint64_t busy_ns;
	uint64_t gpu_ns;
	uint64_t wait_ns;


	uint64_t deadline;

	/* responder results */
	unsigned long drained, served;
	uint64_t *lat_egress;
	int ne;
	int ne_max;
	int r_futex_timeouts;
	int r_errors;
};

static int futex_wait_ns(volatile uint32_t *w, uint32_t val, int ms)
{
	struct timespec ts = {.tv_sec = ms / 1000,
			      .tv_nsec = (long)(ms % 1000) * 1000000};

	return syscall(SYS_futex, (void *)w, FUTEX_WAIT, val, &ts, NULL, 0);
}

static int futex_wake(volatile uint32_t *w)
{
	/* Bump the futex word before waking. The responder reads the word,
	 * re-checks its predicate, then FUTEX_WAITs on that value; if the
	 * word never changes, a wake delivered between the read and the
	 * wait is lost and the waiter blocks for its full timeout. The
	 * increment makes the wait fail with EAGAIN in that race, which is
	 * the standard lost-wakeup guard. */
	__atomic_add_fetch(w, 1, __ATOMIC_RELEASE);
	return syscall(SYS_futex, (void *)w, FUTEX_WAKE, 1, NULL, NULL, 0);
}

static void pin_cpu(int cpu)
{
	cpu_set_t set;

	if (cpu < 0)
		return;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(set), &set);
}

static void *responder(void *arg)
{
	struct daemon *d = (struct daemon *)arg;
	uint32_t published, seen;

	pin_cpu(d->cpu_responder);
	while (bgs_host_mono_ns() < d->deadline) {
		char buf[2048];
		ssize_t r = recv(d->sock, buf, sizeof(buf), MSG_DONTWAIT);

		if (r > 0) {
			d->drained++;
			continue;
		}
		/* The GPU owns completion publication now: it advances
		 * `done_seq` (and `served`) as each request finishes. The
		 * host's `published` is only a retire cursor for the latency
		 * sampler. */
		published = __atomic_load_n(&d->page->done_seq,
					    __ATOMIC_ACQUIRE);
		if (d->page->acked == published) {
			if (d->idle_mode == 1)
				continue; /* busy-poll: re-read done_seq */
			/* Classic futex: read the word, re-check the predicate,
			 * then wait for the value that was seen. The executor
			 * increments the word before waking, so no wake is
			 * lost. */
			seen = __atomic_load_n(&d->page->done_seq,
					       __ATOMIC_ACQUIRE);
			if (d->page->acked !=
			    __atomic_load_n(&d->page->done_seq,
					    __ATOMIC_ACQUIRE))
				continue;
			if (futex_wait_ns(&d->page->done_seq, seen, 20) == -1 &&
			    errno == ETIMEDOUT && d->idle_mode == 2)
				d->r_futex_timeouts++;
			continue;
		}
		while (d->page->acked <
		       __atomic_load_n(&d->page->done_seq, __ATOMIC_ACQUIRE)) {
			uint32_t idx = d->page->acked % BGS_SLOTS;
			struct bgs_done_slot *ds = &d->page->done[idx];
			struct bgs_resp_hdr resp;
			struct sockaddr_in to;
			uint64_t t0;

			if (__atomic_load_n(&ds->state, __ATOMIC_ACQUIRE) !=
				    BGS_DONE ||
			    ds->id != d->page->acked)
				break;
			memset(&resp, 0, sizeof(resp));
			resp.magic = BGS_MAGIC;
			resp.id = ds->id;
			resp.client_ns = ds->client_ns;
			resp.ingress_ns = ds->ingress_ns;
			resp.gpu_start_ns = ds->gpu_start_ns;
			resp.gpu_done_ns = ds->gpu_done_ns;
			memcpy(resp.y, ds->y, sizeof(resp.y));
			memset(&to, 0, sizeof(to));
			to.sin_family = AF_INET;
			to.sin_addr.s_addr = d->page->peers[idx].addr_be;
			to.sin_port = d->page->peers[idx].port_be;
			t0 = bgs_host_mono_ns();
			if (sendto(d->sock, &resp, sizeof(resp), 0,
				   (struct sockaddr *)&to, sizeof(to)) < 0)
				d->r_errors++;
			if (d->ne < d->ne_max)
				d->lat_egress[d->ne++] = bgs_host_mono_ns() - t0;
			/* Recycle: only now may the producer reuse the slot.
			 * The ctl slot must be freed here too, since the GPU
			 * only frees its staged copy, not the mmap'ed page. */
			__atomic_store_n(&d->page->slots[idx].state, BGS_FREE,
					 __ATOMIC_RELEASE);
			__atomic_store_n(&ds->state, BGS_FREE, __ATOMIC_RELEASE);
			d->page->acked++;
			d->served++;
		}
	}
	return NULL;
}

static int cmp_u64(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

	return (x > y) - (x < y);
}

static int rb_sample(void *ctx, void *data, size_t size)
{
	(void)ctx;
	(void)data;
	(void)size;
	return 0;
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

int main(int argc, char **argv)
{
	struct daemon d;
	int port = 39400;
	int cpu_gpu = 2, cpu_responder = 3;
	int idle_mode = 0;
	int ctl_fd;
	struct sockaddr_in addr;
	pthread_t th;
	unsigned long served = 0, drained = 0;

	d.run_seconds = argc > 1 ? atoi(argv[1]) : 5;
	d.work = argc > 2 ? atoi(argv[2]) : 4;
	d.batch = argc > 3 ? atoi(argv[3]) : BGS_MAX_BATCH;
	if (argc > 4)
		cpu_gpu = atoi(argv[4]);
	if (argc > 5)
		cpu_responder = atoi(argv[5]);
	if (argc > 6)
		idle_mode = atoi(argv[6]);
	d.cpu_gpu = cpu_gpu;
	d.cpu_responder = cpu_responder;
	d.idle_mode = idle_mode;
	d.ne = 0;
	d.ne_max = 4000000;
	d.drained = 0;
	d.served = 0;
	d.r_futex_timeouts = 0;
	d.r_errors = 0;
	d.n_busy = 0;
	d.n_idle = 0;
	d.busy_ns = 0;
	d.gpu_ns = 0;
	d.wait_ns = 0;

	d.lat_egress = (uint64_t *)malloc(sizeof(uint64_t) * d.ne_max);

	d.pg = sysconf(_SC_PAGESIZE);

	/* The loader owns the object lifetime and pins the maps; the daemon
	 * only opens them, so both processes share one page and one program. */
	if ((ctl_fd = bpf_obj_get("/sys/fs/bpf/bgs_ctl")) < 0) {
		fprintf(stderr, "bpf_obj_get ctl: %s\n", strerror(errno));
		return 1;
	}
	d.page = (struct bgs_page *)mmap(NULL, BGS_PAGE_MMAP_BYTES,
					PROT_READ | PROT_WRITE, MAP_SHARED,
					ctl_fd, 0);
	if (d.page == MAP_FAILED) {
		perror("mmap ctl");
		return 1;
	}
	{
		int db_fd = bpf_obj_get("/sys/fs/bpf/bgs_db");

		if (db_fd < 0) {
			fprintf(stderr, "bpf_obj_get doorbell: %s\n",
				strerror(errno));
			return 1;
		}
		close(db_fd);
	}

	d.sock = socket(AF_INET, SOCK_DGRAM, 0);
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	{
		/* Bind loopback by default; BGS_BIND lets the responder live on
		 * a veth/NIC so the reply leaves via the same device the tc
		 * ingress hook watches. */
		const char *ba = getenv("BGS_BIND");

		addr.sin_addr.s_addr = ba ? inet_addr(ba)
					  : htonl(INADDR_LOOPBACK);
	}
	addr.sin_port = htons((uint16_t)port);
	if (bind(d.sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("bind");
		return 1;
	}
	{
		int sz = 8 << 20;

		setsockopt(d.sock, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
	}

	/* --- GPU side ------------------------------------------------------ */
	{
		cudaStream_t stream;
		struct bgs_page *d_page;   /* device alias of the BPF control page */
		volatile uint64_t *ts_slot;
		void *d_ts;
		bgs_timer_sync tinfo;
		__u32 consumed = 0;
		uint64_t *lat_dev, *lat_kernel;
		int nb = 0, nk = 0;
		struct ring_buffer *rb;

		lat_dev = (uint64_t *)malloc(sizeof(uint64_t) * 4000000);
		lat_kernel = (uint64_t *)malloc(sizeof(uint64_t) * 4000000);

		cudaSetDevice(0);
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		/* Publish the BPF page to the device: cudaHostRegisterMapped
		 * pins it and cudaHostGetDevicePointer returns a pointer that
		 * aliases the *same* physical pages the eBPF program wrote, so
		 * bgs_exec_kernel reads `slots[]` and writes `done[]` in place.
		 * No per-batch staging copy. */
		{
			cudaError_t rc = cudaHostRegister(d.page, BGS_PAGE_MMAP_BYTES,
						  cudaHostRegisterMapped);

			if (rc != cudaSuccess) {
				fprintf(stderr, "cudaHostRegister(page): %s\n",
					cudaGetErrorString(rc));
				return 1;
			}
		}
		{
			cudaError_t rc = cudaHostGetDevicePointer((void **)&d_page,
							  d.page, 0);

			if (rc != cudaSuccess) {
				fprintf(stderr, "cudaHostGetDevicePointer: %s\n",
					cudaGetErrorString(rc));
				return 1;
			}
		}
		cudaHostAlloc((void **)&ts_slot, 8, cudaHostAllocMapped);
		cudaHostGetDevicePointer(&d_ts, (void *)ts_slot, 0);
		tinfo = bgs_timer_sync_sample(ts_slot, d_ts, stream, 64);
		rb = ring_buffer__new(bpf_obj_get("/sys/fs/bpf/bgs_db"),
				      rb_sample, NULL, NULL);
		if (!rb) {
			fprintf(stderr, "ring_buffer__new failed\n");
			return 1;
		}
		printf("daemon: resident kernel, slots=%d work=%d "
		       "gpu_cpu=%d resp_cpu=%d idle=%d (batch=%d ignored)\n",
		       BGS_SLOTS, d.work, cpu_gpu, cpu_responder,
		       idle_mode, d.batch);
		printf("daemon: globaltimer offset=%lld ns (spread %lld ns)\n",
		       (long long)tinfo.offset, (long long)tinfo.spread);

		d.deadline = bgs_host_mono_ns() +
			     (uint64_t)d.run_seconds * 1000000000ull;
		__atomic_store_n(&d.page->ready, 1, __ATOMIC_RELEASE);
		pthread_create(&th, NULL, responder, &d);
		pin_cpu(cpu_gpu);

		uint64_t t_run0 = bgs_host_mono_ns();

		/* Launch **once**: a single resident kernel spins on the
		 * mmap'ed page for the whole run. There is no per-request
		 * control-plane path left on the host; it only (a) drains the
		 * ingress doorbell, (b) samples the page's completion ring for
		 * the latency histograms, and (c) sets the stop deadline the
		 * kernel polls. */
		__atomic_store_n(&d.page->stop_ns, d.deadline, __ATOMIC_RELEASE);
		bgs_exec_kernel<<<1, 32, 0, stream>>>(d_page, d.work,
						     tinfo.offset);

		{
			uint64_t t_charge = bgs_host_mono_ns();

			for (;;) {
				__u32 seq, head;
				uint64_t now;

				head = __atomic_load_n(&d.page->head,
						       __ATOMIC_ACQUIRE);
				seq = __atomic_load_n(&d.page->done_seq,
						      __ATOMIC_ACQUIRE);
				while (consumed < seq) {
					__u32 idx = consumed % BGS_SLOTS;
					struct bgs_done_slot *ds =
						&d.page->done[idx];

					if (__atomic_load_n(&ds->state,
							    __ATOMIC_ACQUIRE) ==
						    BGS_DONE) {
						if (ds->gpu_done_ns >
							    ds->gpu_start_ns &&
						    nb < 4000000)
							lat_dev[nb++] =
								ds->gpu_done_ns -
								ds->gpu_start_ns;
						if (nk < 4000000 &&
						    ds->ingress_ns &&
						    ds->gpu_done_ns >
							    ds->ingress_ns)
							lat_kernel[nk++] =
								ds->gpu_done_ns -
								ds->ingress_ns;
						if (ds->gpu_done_ns >
						    ds->gpu_start_ns)
							d.gpu_ns +=
								ds->gpu_done_ns -
								ds->gpu_start_ns;
					}
					consumed++;
				}
				__atomic_store_n(&d.page->published, consumed,
						 __ATOMIC_RELEASE);
				ring_buffer__consume(rb);

				now = bgs_host_mono_ns();
				if (consumed < head) {
					d.busy_ns += now - t_charge;
					d.n_busy++;
				} else {
					d.wait_ns += now - t_charge;
					d.n_idle++;
				}
				t_charge = now;
				if (now >= d.deadline)
					break;
			}
		}
		/* Let the resident kernel retire whatever is still queued:
		 * clear its stop clock, wait briefly for the ring to drain,
		 * then set the clock so it can exit. */
		__atomic_store_n(&d.page->stop_ns, 0, __ATOMIC_RELEASE);
		{
			uint64_t drain_deadline =
				bgs_host_mono_ns() + 2000000000ull;

			for (;;) {
				__u32 seq = __atomic_load_n(&d.page->done_seq,
							    __ATOMIC_ACQUIRE);
				__u32 head = __atomic_load_n(&d.page->head,
							     __ATOMIC_ACQUIRE);

				if (seq >= head ||
				    bgs_host_mono_ns() > drain_deadline)
					break;
			}
		}
		__atomic_store_n(&d.page->stop_ns, bgs_host_mono_ns(),
				 __ATOMIC_RELEASE);
		cudaStreamSynchronize(stream);   /* resident kernel exits */
		d.deadline = 0; /* let the responder exit immediately */
		futex_wake(&d.page->done_seq);
		pthread_join(th, NULL);
		drained = d.drained;
		served = d.served;

		{
			double run_ns = (double)(bgs_host_mono_ns() - t_run0);

			printf("daemon: consumed=%u published=%u head=%u "
			       "drops=%u acked=%u\n",
			       consumed, d.page->published, d.page->head,
			       d.page->drops, d.page->acked);
			printf("daemon: drained=%lu served=%lu errors=%d "
			       "futex_timeouts=%d\n",
			       drained, served, d.r_errors,
			       d.r_futex_timeouts);
			printf("daemon: gpu batches busy=%lu idle_polls=%lu "
			       "(busy_wall %.1f%% of %.2f s; wall %.0f req/s)\n",
			       d.n_busy, d.n_idle,
			       (d.busy_ns + d.wait_ns) > 0
				       ? 100.0 * (double)d.busy_ns /
						 (double)(d.busy_ns + d.wait_ns)
				       : 0.0,
			       run_ns / 1e9,
			       run_ns > 0 ? (double)served * 1e9 / run_ns : 0.0);
			/* Occupancy: GPU kernel time over host wall time while
			 * a request was in flight. The shortfall is the gap
			 * between one slot's arithmetic finishing and the next
			 * request being visible to the resident kernel (ingress
			 * + doorbell latency), plus spin-loop polling cost —
			 * not a per-request host launch. */
			printf("daemon: gpu kernel %.2f s busy_wall %.2f s "
			       "wait %.2f s -> occupancy %.1f%% "
			       "(bubble %.2f s)\n",
			       (double)d.gpu_ns / 1e9,
			       (double)d.busy_ns / 1e9,
			       (double)d.wait_ns / 1e9,
			       d.busy_ns > 0 ? 100.0 * (double)d.gpu_ns /
						       (double)d.busy_ns
					     : 0.0,
			       (double)(d.busy_ns - d.gpu_ns) / 1e9);
		}
		report("GPU start -> GPU done", lat_dev, nb);
		report("kernel parse -> GPU done", lat_kernel, nk);
		report("responder sendto()", d.lat_egress, d.ne);
	}
	return 0;
}
