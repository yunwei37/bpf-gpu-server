/* SPDX-License-Identifier: Apache-2.0
 * Experimental kernel-owned transport; explicit targets, not selected by default.
 * Existing model-page ABI is unchanged; these are CPU-only registered bytes.
 */
#ifndef BGS_KERNEL_TCP_STATE_H
#define BGS_KERNEL_TCP_STATE_H
#include "bpf_gpu_server_queue.h"
#define BGS_KTCP_RX_STRIDE 4160u
#define BGS_KTCP_TX_STRIDE 512u
struct bgs_ktcp_layout {
    __u64 user_base;
    __u32 capacity, entries, ring_bytes, mem_bytes;
    __u32 conns, buffers, tx, tick;
    __u32 sh, st, ch, ct, cqes, dropped, overflow;
    int listener;
};
struct bgs_ktcp_conn {
    int fd, next, reply, last;
    __u32 receiving, closing, eof, dead, generation;
};
struct bgs_ktcp_request {
    int conn, next;
    __u32 seq, ng, sent, header, header_ready, active, busy, final;
    __u64 tag, client_ns;
    __u32 sends, early, bytes;
};
struct bgs_ktcp_record {
    __u32 seq, gen, sends, early, bytes, aborted;
    __u64 client_ns;
};
struct bgs_ktcp_shared {
    struct bgs_uring_loop_control loop;
    __u32 accepted, received, sends, bytes, completed, closed, aborted;
    __u32 max_connections, max_requests, pending, error;
    __u32 head, reserved, dropped, overflow, sqhead, sqtail, cqhead, cqtail;
    __u32 trace_count, trace_overflow;
    int free, active, accepting;
    __u32 connections, seen, next_final;
    __u32 stopping, cancelled, cancel_done;
    __u32 timeout_active, timeout_removing, timeout_idle, timeout_updating;
    struct bgs_ktcp_request requests[BGS_LLM_SLOTS];
    struct bgs_ktcp_record records[128];
};
#endif
