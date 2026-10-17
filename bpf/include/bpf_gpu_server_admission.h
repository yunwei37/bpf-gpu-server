/* SPDX-License-Identifier: Apache-2.0 */
#ifndef BPF_GPU_SERVER_ADMISSION_H
#define BPF_GPU_SERVER_ADMISSION_H
#include "bpf_gpu_server_queue.h"
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__uint(map_flags, BPF_F_MMAPABLE);
	__type(key, __u32);
	__type(value, struct bgs_page);
} ctl SEC(".maps");


/* Loader discovers capacity from the existing listener backlog; the loop
 * reuses that exact map rather than creating a second admission queue. */
struct {
    __uint(type, BPF_MAP_TYPE_QUEUE);
    __uint(max_entries, 1);
    __type(value, struct bgs_llm_pending);
} pending SEC(".maps");
/* Reserve the sequence and physical slot in one atomic operation. Separate
 * state/head CAS can reject another producer's transient WRITING reservation.
 * The bitmap also prevents wrap over a producer paused before publishing state. */
static __always_inline struct bgs_llm_slot *llm_reserve(struct bgs_page *p)
{
    if (*(volatile __u64 *)&p->stop_ns) return 0;
    for (__u32 attempt=0; attempt<BGS_LLM_SLOTS; attempt++) {
        __u64 word=*(volatile __u64 *)&p->llm_admission;
        __u32 head=(__u32)word, idx=head%BGS_LLM_SLOTS;
        __u32 reserved=word>>32, bit=1u<<idx;
        if (reserved&bit) break;
        __u64 next=((__u64)(reserved|bit)<<32)|(__u32)(head+1);
        if (__sync_val_compare_and_swap(&p->llm_admission,word,next)==word) {
            struct bgs_llm_slot *slot=&p->llm[idx];
            slot->state=BGS_WRITING;
            return slot;
        }
    }
    return 0;
}


static __always_inline void llm_publish(struct bgs_llm_slot *slot,
                                       const struct bgs_llm_pending *request)
{
    for (__u32 i=0;i<BGS_LLM_MAX_TOK;i++) slot->tok_in[i]=request->tokens[i];
    slot->n_prompt=request->n_prompt; slot->n_gen=request->n_gen;
    slot->produced=0; slot->pad=request->flags;
    slot->addr_be=request->addr_be; slot->port_be=request->port_be;
    slot->client_ns=request->client_ns; slot->ingress_ns=request->ingress_ns;
    slot->gpu_done_ns=0; slot->state=BGS_PENDING;
}
#endif
