/* SPDX-License-Identifier: Apache-2.0
 * bpf-gpu-server: canonical layout of the kernel<->GPU control page.
 *
 * Included by the eBPF ingress program, the resident GPU executor and the
 * gateway. POD only. The page is a single `BPF_MAP_TYPE_ARRAY` entry with
 * `BPF_F_MMAPABLE`, so the very same bytes are written by the BPF program in
 * softirq context and read/written by both userspace processes over the same
 * mapping.
 *
 * Ownership (one writer per field, no locks):
 *   slots[i].state    producer: FREE -> PENDING. GPU: PENDING -> FREE once the
 *                     request is in registers.
 *   done[i].state     executor: FREE -> DONE after done[i] is complete.
 *                     gateway:  DONE -> FREE once the response is on the wire.
 *   head              producer: requests published so far.
 *   done_seq          executor: futex word, bumped after each batch.
 *   published         executor: completions made visible (== head consumed).
 *   acked             gateway: responses sent (== completions consumed).
 *   ready             executor: 1 once calibrated and looping.
 *   drops             producer: requests that found the slot still busy.
 *
 * A slot is recyclable only when both `slots[i].state` and `done[i].state` are
 * FREE, so a completion cannot be overwritten before the gateway has sent it.
 *
 * The payload is exactly what the client sent: a 64-byte float vector plus its
 * own monotonic timestamp, echoed back in the response, so the client measures
 * a round trip without trusting server clocks.
 */
#ifndef BPF_GPU_SERVER_QUEUE_H
#define BPF_GPU_SERVER_QUEUE_H

#ifndef __VMLINUX_H__
typedef unsigned int __u32;
typedef unsigned long long __u64;
typedef unsigned short __u16;
#endif

#define BGS_PORT 39400

/* LLM request ring: a second, independent slot ring that carries token ids
 * instead of a float vector. The BPF producer fills `tok_in` from the packet;
 * the resident model executor generates into `tok_out`, bumping `produced`
 * after each token so a client can measure TTFT (first token) and TPOT
 * (inter-token) over the same kernel->page path. */
#define BGS_LLM_PORT 39402
#define BGS_LLM_TCP_PORT 39403 /* same LLM ring, over stream TCP */
#define BGS_LLM_MAGIC 0x514c4d51u /* 'Q','M','L','Q' little-endian */
#ifndef BGS_LLM_SLOTS
#define BGS_LLM_SLOTS 8
#endif
#define BGS_LLM_MAX_TOK 64
/* Conservative peer buckets, not a connection/admission capacity limit. */
#define BGS_PENDING_BUCKETS 4096
#define BGS_LLM_TRANSPORT_MASK 3u
#define BGS_LLM_VALIDATING 4u /* stream request awaits model-vocabulary check */
#define BGS_LLM_REJECTED 8u /* executor rejected request before producing tokens */


#define BGS_SLOTS 64
#define BGS_MAX_BATCH 16
#define BGS_VEC 16

/* Bootstrap correlation of kernel io_uring events with the model page. */
struct bgs_uring_context {
    __u32 pid, pad;
    __u64 ring, nsdev, nsino;
};
/* io_uring registered-memory prefix used by its model-agnostic BPF loop. */
struct bgs_uring_loop_control {
    __u32 sq_head, sq_tail, cq_tail;
    __u32 model_quiesced, finished, stop_requested;
    __u64 deadline_ns;
    __u32 cq_seen, timer_may_idle;
    __u64 now_ns;
    __u32 admission_seen, backlog_peak;
};
#define BGS_IO_SEND_TAG 0xbff0ULL
#define BGS_IO_ABORT_TAG 0xbff1ULL

/* CPU-owned admission backlog; model slots keep their existing layout. */
struct bgs_llm_pending {
    __u32 n_prompt, n_gen, addr_be;
    __u16 port_be, flags;
    __u64 client_ns, ingress_ns;
    __u32 tokens[BGS_LLM_MAX_TOK];
};

struct bgs_llm_slot {
	__u32 state;      /* producer FREE->PENDING; executor ->DONE; TX ->FREE */
	__u32 n_gen;      /* tokens requested */
	__u32 n_prompt;   /* prompt tokens written to tok_in */
	__u32 produced;   /* tokens written to tok_out so far (executor) */
	__u32 addr_be;    /* client peer, network order */
	__u16 port_be;
	__u16 pad;       /* transport 1=TCP, 2=HTTP; producer VALIDATING, GPU status */
	__u64 client_ns;
	__u64 ingress_ns;
	__u64 gpu_done_ns;
	__u32 tok_in[BGS_LLM_MAX_TOK];
	__u32 tok_out[BGS_LLM_MAX_TOK];
};

/* Wire layout of a request datagram. */
#define BGS_MAGIC 0x46504251u /* 'Q','B','P','F' little-endian */
struct bgs_req_hdr {
	__u32 magic;
	__u32 pad;
	__u64 client_ns;
	float x[BGS_VEC];
};

/* Wire layout of a response datagram. */
struct bgs_resp_hdr {
	__u32 magic;
	__u32 id;
	__u32 pad;
	__u32 pad2;
	__u64 client_ns;  /* echoed from the request */
	__u64 ingress_ns; /* kernel parse time, CLOCK_MONOTONIC */
	__u64 gpu_start_ns;
	__u64 gpu_done_ns;
	float y[BGS_VEC];
};

enum bgs_state {
	BGS_FREE = 0,
	BGS_PENDING = 1,
	BGS_DONE = 2,
	BGS_WRITING = 3, /* producer reservation, before publishing PENDING */
};

struct bgs_ctl_slot {
	__u32 state;
	__u32 id;
	__u64 client_ns;
	__u64 ingress_ns; /* bpf_ktime_get_ns() / CLOCK_MONOTONIC */
	float x[BGS_VEC];
};

struct bgs_done_slot {
	__u32 id;
	__u32 state;
	__u64 client_ns; /* copied from the request: the responder echoes it */
	__u64 ingress_ns;
	__u64 gpu_start_ns; /* %globaltimer */
	__u64 gpu_done_ns;  /* %globaltimer */
	float y[BGS_VEC];
};

/* Where to send the answer, recorded by the producer for each slot. */
struct bgs_peer {
	__u32 addr_be; /* IPv4, network order */
	__u16 port_be;
	__u16 pad2;
};
struct bgs_page {
	__u32 head;
	__u32 done_seq;
	__u32 published;
	__u32 acked;
	__u32 drops;
	__u32 ready;
	__u32 n_slots;
	__u32 served;      /* completions published to done[] (GPU-owned) */
	__u32 pad0;
	__u64 stop_ns;     /* host CLOCK_MONOTONIC deadline for the resident
			    * kernel; 0 means "not set". */

	struct bgs_ctl_slot slots[BGS_SLOTS];
	struct bgs_peer peers[BGS_SLOTS];
	struct bgs_done_slot done[BGS_SLOTS];

	/* LLM request ring (see bgs_llm_slot above). Kept at the tail so the
	 * MLP offsets are unchanged. */
	/* One CPU atomic reservation advances head and owns its physical slot.
	 * The high-word bit stays set until TX has finished all slot accesses;
	 * it protects wrap even if the producer has not published WRITING yet. */
	union {
		struct { __u32 llm_head, llm_reserved; };
		__u64 llm_admission;
	};
	struct bgs_llm_slot llm[BGS_LLM_SLOTS];
    __u32 llm_queued, admission_pad; /* CPU atomic count, read-only on GPU */
    __u32 llm_peer_pending[BGS_PENDING_BUCKETS];
};

/* Network-order peer identity is identical in skb, slots and sockaddr_in.
 * Collisions retain another peer longer; they never permit an early close. */
#ifdef __CUDACC__
static __host__ __device__ __forceinline__
#else
static __inline__ __attribute__((always_inline))
#endif
__u32 bgs_pending_bucket(__u32 addr_be, __u16 port_be)
{
    __u32 word=addr_be ^ (((__u32)port_be<<16)|port_be);
    word^=word>>16;word*=0x7feb352du;word^=word>>15;
    word*=0x846ca68bu;word^=word>>16;
    return word & (BGS_PENDING_BUCKETS-1);
}

#define BGS_PAGE_BYTES ((int)sizeof(struct bgs_page))

/* A BPF_F_MMAPABLE array map exposes its value at offset 0 and the region is
 * the value size rounded up to a page multiple. Mapping only BGS_PAGE_BYTES
 * (or a fixed two pages) is wrong: the layout grows with slot count and
 * transport metadata. A short mapping leaves fields outside the valid region. */
#define BGS_PAGE_MMAP_BYTES \
	((long)((((long)BGS_PAGE_BYTES + 4095L) / 4096L) * 4096L))

#endif /* BPF_GPU_SERVER_QUEUE_H */
