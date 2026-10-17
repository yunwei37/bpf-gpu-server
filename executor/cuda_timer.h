// SPDX-License-Identifier: Apache-2.0
// bpf-gpu-server: host<->device timestamp correlation.
//
// The executor stamps completions with `%globaltimer`. On this host that
// counter ticks at the same 1 GHz rate as CLOCK_MONOTONIC (measured:
// host_delta / gpu_delta = 0.99988..0.99999 over ~0.69 s), so a single
// constant offset relates the two.
//
// The offset is calibrated by having the GPU write `%globaltimer` straight
// into pinned, zero-copy memory and having the host spin on that location. The
// host's read is therefore separated from the device's store only by the
// memory-write latency, not by a launch/synchronize round trip. Taking the
// minimum over trials (the read that was closest to the store) leaves a bias of
// the same order as one PCIe write, reported back as `spread`.
#ifndef BPF_GPU_SERVER_CUDA_TIMER_H
#define BPF_GPU_SERVER_CUDA_TIMER_H

#include <cuda_runtime.h>
#include <cstdint>
#include <ctime>

struct bgs_timer_sync {
	uint64_t offset; /* host_mono_ns - gpu_globaltimer_ns */
	int64_t spread;  /* max-min over accepted samples, ns */
	int valid;
};

__global__ void bgs_ping_globaltimer_kernel(unsigned long long *out)
{
	unsigned long long t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	if (threadIdx.x == 0) {
		__threadfence_system();
		*out = t;
		__threadfence_system();
	}
}

__device__ __forceinline__ unsigned long long bgs_globaltimer(void)
{
	unsigned long long t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

static inline uint64_t bgs_host_mono_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// `host_slot` must be pinned+mapped (cudaHostAllocMapped) and `dev_slot` its
// device pointer. `stream` may be any stream. Returns the best offset estimate
// and the spread of the accepted samples.
static inline bgs_timer_sync bgs_timer_sync_sample(volatile uint64_t *host_slot,
						 void *dev_slot,
						 cudaStream_t stream, int iters)
{
	bgs_timer_sync r = {0, 0, 0};
	int64_t mn = 0, mx = 0;
	int accepted = 0;

	if (iters < 2)
		iters = 2;
	for (int i = 0; i < iters; i++) {
		uint64_t gpu, host_now;
		int64_t d;
		int spins = 0;

		*host_slot = 0;
		__sync_synchronize();
		bgs_ping_globaltimer_kernel<<<1, 32, 0, stream>>>(
			(unsigned long long *)dev_slot);
		while (*host_slot == 0 && spins < 100000000)
			spins++;
		host_now = bgs_host_mono_ns();
		gpu = *host_slot;
		if (gpu == 0)
			continue;
		d = (int64_t)(host_now - gpu);
		if (accepted == 0) {
			mn = mx = d;
		} else {
			if (d < mn)
				mn = d;
			if (d > mx)
				mx = d;
		}
		accepted++;
		cudaStreamSynchronize(stream);
	}
	if (!accepted) {
		r.valid = 0;
		return r;
	}
	r.offset = (uint64_t)mn;
	r.spread = mx - mn;
	r.valid = 1;
	return r;
}

#endif /* BPF_GPU_SERVER_CUDA_TIMER_H */
