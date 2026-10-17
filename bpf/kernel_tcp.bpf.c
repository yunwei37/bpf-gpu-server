// SPDX-License-Identifier: GPL-2.0
// Experimental transport: the existing BPF loop owns TCP progression.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/errno.h>
#include "bpf_gpu_server_admission.h"
#include "kernel_tcp_state.h"

/* Stable io_uring UAPI flag values, to be checked against actual build headers. */
#define KTCP_CQE_MORE (1u << 1)
#define KTCP_CANCEL_ALL (1u << 0)
#define KTCP_CANCEL_ANY (1u << 2)
#define KTCP_TIMEOUT_UPDATE (1u << 1)
#define KTCP_MSG_NOSIGNAL 0x4000u

extern __u8 *bpf_io_uring_get_region(struct iou_ctx *, __u32, const size_t) __ksym;
extern int bpf_io_uring_submit_sqes(struct iou_ctx *, __u32) __ksym;
/* Derived once from the actual existing stream map, ring setup and allocation;
 * readonly specialization is required for verifier memory bounds, not a knob. */
const volatile struct bgs_ktcp_layout bgs_ktcp_config = {};
static const char reply_ok[] =
    "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: ";
static const char reply_error[] =
    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/octet-stream\r\nContent-Length: ";

struct step {
    __u8 *memory, *rings, *sq;
    struct bgs_ktcp_shared *s;
    struct bgs_page *page;
    __u32 sh, st, cqhead;
};
static __always_inline void order_memory(void)
{
    asm volatile("" ::: "memory");
}
static __always_inline __u32 load32(const __u32 *p)
{
    __u32 value = *(volatile const __u32 *)p;
    order_memory();
    return value;
}
static __always_inline struct bgs_ktcp_conn *connection(struct step *x, __u32 i)
{
    __u64 index = i;
    /* Keep the zero-extended index and its checked pointer arithmetic identical
     * for verifier tracking, even when CQ user_data has nonzero upper bits. */
    asm volatile("%0 <<= 32; %0 >>= 32" : "+r"(index));
    if (index >= bgs_ktcp_config.capacity) return 0;
    return (void *)(x->memory + bgs_ktcp_config.conns + index * sizeof(struct bgs_ktcp_conn));
}
static __always_inline __u64 operation_id(__u32 kind, __u32 value, __u32 generation)
{
    return ((__u64)kind << 56) | ((__u64)(generation & 0xffffffu) << 32) | value;
}
static __always_inline int enqueue(struct step *x, const struct io_uring_sqe *entry)
{
    if (x->st - x->sh >= bgs_ktcp_config.entries) return 0;
    __u32 i = x->st & (bgs_ktcp_config.entries - 1);
    struct io_uring_sqe *sqe = (void *)(x->sq + i * sizeof(*sqe));
    __builtin_memcpy(sqe, entry, sizeof(*sqe));
    x->st++;
    x->s->pending++;
    return 1;
}
static __always_inline void publish(struct step *x)
{
    order_memory();
    *(volatile __u32 *)(x->rings + bgs_ktcp_config.st) = x->st;
}
static __noinline int admit(struct bgs_page *page)
{
    struct bgs_llm_pending request;
    for (__u32 n = 0; n < BGS_LLM_SLOTS; n++) {
        if (bpf_map_peek_elem(&pending, &request)) break;
        struct bgs_llm_slot *slot = llm_reserve(page);
        if (!slot) break;
        if (bpf_map_pop_elem(&pending, &request)) return -1;
        llm_publish(slot, &request);
        __sync_fetch_and_sub(&page->llm_queued, 1);
        __sync_fetch_and_sub(&page->llm_peer_pending[
            bgs_pending_bucket(request.addr_be, request.port_be)], 1);
    }
    return 0;
}
static __always_inline int retire(struct step *x, __u32 i, int aborted)
{
    struct bgs_ktcp_shared *s = x->s;
    if (i >= BGS_LLM_SLOTS) return -1;
    struct bgs_ktcp_request *r = &s->requests[i];
    if (r->conn >= 0) {
        struct bgs_ktcp_conn *c = connection(x, (__u32)r->conn);
        if (!c) return -1;
        c->reply = r->next;
        if (c->reply < 0) c->last = -1;
    }
    __u32 record = s->trace_count++;
    if (record < 128) {
        s->records[record] = (struct bgs_ktcp_record){
            r->seq, r->ng, r->sends, r->early, r->bytes,
            (__u32)aborted, r->client_ns
        };
    } else s->trace_overflow++;
    s->completed++;
    if (aborted) s->aborted++;
    r->active = 0;
    return 0;
}
static __noinline int drain_entry(struct step *x, __u32 end)
{
    struct bgs_ktcp_shared *s=x->s;

        if (x->cqhead == end) return 1;
        __u32 j = x->cqhead & (2 * bgs_ktcp_config.entries - 1);
        struct io_uring_cqe *cq = (void *)(x->rings + bgs_ktcp_config.cqes +
                                         j * sizeof(struct io_uring_cqe));
        struct io_uring_cqe e;
        __builtin_memcpy(&e, cq, sizeof(e));
        x->cqhead++;
        __u32 kind = e.user_data >> 56, index = (__u32)e.user_data;
        if (kind == 8 && (e.flags & KTCP_CQE_MORE)) return 0;
        if (!s->pending) return -1;
        s->pending--;
        __u32 prefix = e.user_data >> 48;
        if (prefix == BGS_IO_SEND_TAG || prefix == BGS_IO_ABORT_TAG) kind = 7;
        if (kind == 1) {
            struct bgs_ktcp_conn *c = connection(x, index);
            if (!c) return -1;
            s->accepting = -1;
            if (e.res >= 0) {
                c->fd = e.res; c->next = s->active; s->active = index;
                s->connections++; s->accepted++;
                if (s->connections > s->max_connections)
                    s->max_connections = s->connections;
                if (s->stopping) c->dead = 1;
            } else {
                c->next = s->free; s->free = index;
                if (e.res != -ECANCELED) return -1;
            }
        } else if (kind == 2) {
            struct bgs_ktcp_conn *c = connection(x, index);
            if (!c) return -1;
            c->receiving = 0;
            if (e.res > 0) s->received++;
            else if (e.res == 0) c->eof = 1;
            else if (e.res != -EAGAIN && e.res != -EINTR) c->dead = 1;
        } else if (kind == 3 || kind == 7) {
            __u32 i = index % BGS_LLM_SLOTS;
            struct bgs_ktcp_request *r = &s->requests[i];
            if (!r->active || !r->busy || r->tag != e.user_data) return -1;
            r->busy = 0;
            if (prefix == BGS_IO_ABORT_TAG) {
                if (e.res || r->seq != s->next_final || retire(x, i, 1)) return -1;
                s->next_final++;
            } else if (e.res > 0) {
                r->sent += e.res; r->bytes += e.res; s->bytes += e.res;
                r->sends++; s->sends++; if (!r->final) r->early++;
                __u32 length = (e.user_data >> 32) & 0xffff;
                if (kind == 7 && (__u32)e.res == length) {
                    if (r->seq != s->next_final || retire(x, i, 0)) return -1;
                    s->next_final++;
                }
            } else if (r->conn >= 0) {
                struct bgs_ktcp_conn *c = connection(x, (__u32)r->conn);
                if (!c) return -1;
                c->dead = 1;
            }
        } else if (kind == 4) {
            struct bgs_ktcp_conn *c = connection(x, index);
            if (!c) return -1;
            c->fd = -1; c->closing = 0; s->closed++;
        } else if (kind == 5) s->cancel_done = 1;
        else if (kind == 6) { /* target RECV retains its own CQE */ }
        else if (kind == 8) {
            s->timeout_active = 0;
            if (e.res != -ECANCELED) return -1;
        } else if (kind == 9) {
            if (e.res == -ENOENT) s->timeout_removing = 0;
            else if (e.res < 0) return -1;
        } else if (kind == 10) {
            s->timeout_updating = 0;
            if (!e.res) s->timeout_idle = index;
            s->loop.timer_may_idle = s->timeout_idle;
            if (e.res < 0 && e.res != -ENOENT && e.res != -EALREADY &&
                !(s->stopping && e.res == -ECANCELED)) return -1;
        } else return -1;
    
    return 0;
}
struct drain_context { struct step *x; __u32 end; int failed; };
static long drain_callback(__u64 index, void *opaque)
{
    struct drain_context *context=opaque;
    int result=drain_entry(context->x,context->end);
    if(result<0)context->failed=1;
    return result!=0;
}
static __noinline int drain_completions(struct step *x)
{
    struct drain_context context={.x=x,.end=load32((void *)(x->rings+bgs_ktcp_config.ct))};
    long result=bpf_loop(2*bgs_ktcp_config.entries,drain_callback,&context,0);
    if(result<0||context.failed)return -1;
    order_memory();
    *(volatile __u32 *)(x->rings+bgs_ktcp_config.ch)=x->cqhead;
    return 0;
}
struct peer_context {
    struct step *x;
    __u32 addr_be;
    __u16 port_be;
    int peer,found,failed;
};
static long peer_callback(__u64 index, void *opaque)
{
    struct peer_context *context=opaque;
    if(context->peer<0)return 1;
    struct step *x=context->x;
    __u64 peer=(__u32)context->peer;
    asm volatile("%0 <<= 32; %0 >>= 32" : "+r"(peer));
    if(peer>=bgs_ktcp_config.capacity){context->failed=1;return 1;}
    struct bgs_ktcp_conn *c=connection(x,peer);
    if(!c){context->failed=1;return 1;}
    struct sockaddr_in *addr=(void *)(x->memory+bgs_ktcp_config.buffers+
                         peer*BGS_KTCP_RX_STRIDE+4096);
    if(context->addr_be==addr->sin_addr.s_addr&&context->port_be==addr->sin_port){
        context->found=1;return 1;
    }
    context->peer=c->next;
    return 0;
}
static __noinline int claim_requests(struct step *x)
{
    struct bgs_ktcp_shared *s = x->s;
    for (__u32 n = 0; n < BGS_LLM_SLOTS; n++) {
        if (s->seen == load32(&x->page->llm_head)) break;
        __u32 i = s->seen % BGS_LLM_SLOTS;
        struct bgs_ktcp_request *r = &s->requests[i];
        struct bgs_llm_slot *slot = &x->page->llm[i];
        __u32 state = load32(&slot->state);
        if (r->active || (state != BGS_PENDING && state != BGS_DONE)) break;
        struct peer_context context={.x=x,.addr_be=slot->addr_be,
                          .port_be=slot->port_be,.peer=s->active};
        long visited=bpf_loop(bgs_ktcp_config.capacity,peer_callback,&context,0);
        if(visited<0||context.failed)return -1;
        int peer=context.found?context.peer:-1;
        if (peer < 0 && !s->stopping) break;
        *r = (struct bgs_ktcp_request){
            .conn = peer, .next = -1, .seq = s->seen,
            .ng = load32(&slot->n_gen), .active = 1,
            .client_ns = *(volatile __u64 *)&slot->client_ns
        };
        if (peer >= 0) {
            struct bgs_ktcp_conn *c = connection(x, (__u32)peer);
            if (!c) return -1;
            if (c->last >= 0) {
                if ((__u32)c->last >= BGS_LLM_SLOTS) return -1;
                s->requests[c->last].next = i;
            } else c->reply = i;
            c->last = i;
        }
        s->seen++;
    }
    return 0;
}
static __noinline int prepare_header(struct step *x, __u32 i, __u32 bad, __u32 flags)
{
    if (i >= BGS_LLM_SLOTS) return -1;
    struct bgs_ktcp_request *r = &x->s->requests[i];
    __u8 *tx = x->memory + bgs_ktcp_config.tx + i * BGS_KTCP_TX_STRIDE;
    __u32 count = 0;
    if ((flags & BGS_LLM_TRANSPORT_MASK) == 2) {
        if (bad) {
            __builtin_memcpy(tx, reply_error, sizeof(reply_error)-1);
            count = sizeof(reply_error)-1;
        } else {
            __builtin_memcpy(tx, reply_ok, sizeof(reply_ok)-1);
            count = sizeof(reply_ok)-1;
        }
        __u32 value = bad ? 0 : r->ng * 4;
        if (value > BGS_LLM_MAX_TOK * 4) return -1;
        char digits[4] = {};
        __u32 n = 0;
        for (__u32 k = 0; k < 4; k++) {
            digits[n++] = '0' + value % 10; value /= 10;
            if (!value) break;
        }
        for (__u32 k = 0; k < 4; k++) {
            if (!n) break;
            tx[count++] = digits[--n];
        }
        tx[count++] = '\r'; tx[count++] = '\n';
        tx[count++] = '\r'; tx[count++] = '\n';
    }
    r->header = count; r->header_ready = 1;
    return 0;
}
struct token_context {
    __u8 *tx;
    struct bgs_llm_slot *slot;
    __u32 header,first,produced;
};
static long token_callback(__u64 index, void *data)
{
    struct token_context *context = data;
    __u32 header = context->header;
    __u32 produced = context->produced;
    if (header > BGS_KTCP_TX_STRIDE - 4 * BGS_LLM_MAX_TOK ||
        produced > BGS_LLM_MAX_TOK || index >= BGS_LLM_MAX_TOK ||
        index >= produced) return 1;
    if (index < context->first) return 0;
    __u32 j = (__u32)index;
    __u32 token = load32(&context->slot->tok_out[j]);
    for (__u32 byte = 0; byte < 4; byte++)
        context->tx[header+j*4+byte] = (token >> (8*byte)) & 255;
    return 0;
}
static __noinline int send_one(struct step *x, __u32 i)
{
    if (i >= BGS_LLM_SLOTS) return -1;
    struct bgs_ktcp_shared *s = x->s;
        struct bgs_ktcp_request *r = &s->requests[i];
        if (!r->active) return 0;
        if (r->busy) return 1;
        struct bgs_llm_slot *slot = &x->page->llm[i];
        __u32 state = load32(&slot->state);
        struct bgs_ktcp_conn *c = 0;
        if (r->conn >= 0) {
            c = connection(x, (__u32)r->conn);
            if (!c) return -1;
        }
        int gone = !c || c->dead || s->stopping;
        if (gone) {
            if (r->seq != s->next_final) return 1;
            if (state != BGS_DONE) {
                if (!s->stopping || !load32(&s->loop.model_quiesced)) return 1;
                order_memory(); *(volatile __u32 *)&slot->state = BGS_DONE;
            }
            struct io_uring_sqe entry = {};
            entry.opcode = IORING_OP_NOP;
            entry.user_data = (BGS_IO_ABORT_TAG << 48) | r->seq;
            if (enqueue(x, &entry)) {
                r->busy = 1; r->tag = entry.user_data; r->final = 1;
            }
            return 1;
        }
        if (c->reply != (int)i) return 1;
        __u32 flags = *(volatile __u16 *)&slot->pad;
        if (flags & BGS_LLM_VALIDATING) return 1;
        __u32 bad = flags & BGS_LLM_REJECTED;
        if (bad && (state != BGS_DONE || r->seq != s->next_final)) return 1;
        if (bad && (flags & BGS_LLM_TRANSPORT_MASK) == 1) {
            c->dead = 1; return 1;
        }
        if (!r->header_ready && prepare_header(x, i, bad, flags)) return -1;
        __u32 header = r->header;
        if (header > BGS_KTCP_TX_STRIDE - 4 * BGS_LLM_MAX_TOK) return -1;
        __u32 produced = bad ? 0 : load32(&slot->produced);
        if (produced > BGS_LLM_MAX_TOK) return -1;
        if ((state != BGS_DONE || r->seq != s->next_final) &&
            produced && produced == r->ng) produced--;
        int final = state == BGS_DONE && r->seq == s->next_final &&
                    (bad || produced == r->ng);
        if (!produced && !bad) return 1;
        __u32 total = header + produced * 4;
        if (total <= r->sent) return 1;
        __u32 first = r->sent > header ? (r->sent-header)/4 : 0;
        __u8 *tx = x->memory + bgs_ktcp_config.tx + i * BGS_KTCP_TX_STRIDE;
        struct token_context tokens = {
            .tx=tx, .slot=slot, .header=header, .first=first, .produced=produced
        };
        if (bpf_loop(BGS_LLM_MAX_TOK, token_callback, &tokens, 0) < 0)
            return -1;
        struct io_uring_sqe entry = {};
        entry.opcode = IORING_OP_SEND; entry.fd = c->fd;
        entry.addr = bgs_ktcp_config.user_base + bgs_ktcp_config.tx +
                     i * BGS_KTCP_TX_STRIDE + r->sent;
        entry.len = total-r->sent; entry.msg_flags = KTCP_MSG_NOSIGNAL;
        entry.user_data = final ? ((BGS_IO_SEND_TAG << 48) |
            ((__u64)entry.len << 32) | r->seq) : operation_id(3, r->seq, 0);
        if (enqueue(x, &entry)) {
            r->busy = 1; r->tag = entry.user_data; r->final = final;
        }
    return 1;
}
struct reply_context { struct step *x; __u32 active; int failed; };
static long reply_callback(__u64 index, void *data)
{
    struct reply_context *context = data;
    if (index >= BGS_LLM_SLOTS) return 1;
    int result = send_one(context->x, (__u32)index);
    if (result < 0) { context->failed = 1; return 1; }
    context->active += result;
    return 0;
}
static __noinline int send_replies(struct step *x)
{
    struct reply_context context = { .x = x };
    if (bpf_loop(BGS_LLM_SLOTS, reply_callback, &context, 0) < 0 ||
        context.failed) return -1;
    if (context.active > x->s->max_requests)
        x->s->max_requests = context.active;
    return context.active;
}
struct connection_context { struct step *x; int previous,current,failed; };
static __noinline int progress_connection(struct connection_context *context)
{
    if(context->current<0)return 1;
    struct step *x=context->x;
    struct bgs_ktcp_shared *s=x->s;

        __u64 i = (__u32)context->current;
        asm volatile("%0 <<= 32; %0 >>= 32" : "+r"(i));
        if (i >= bgs_ktcp_config.capacity) return -1;
        struct bgs_ktcp_conn *c = connection(x, i);
        if (!c) return -1;
        if (c->fd < 0 && !c->closing) {
            int next = c->next;
            if (context->previous < 0) s->active = next;
            else {
                struct bgs_ktcp_conn *p = connection(x, (__u32)context->previous);
                if (!p) return -1;
                p->next = next;
            }
            c->next = s->free; s->free = i; s->connections--;
            context->current = next; return 0;
        }
        if (s->stopping) c->dead = 1;
        if (c->dead && c->receiving && !c->closing) {
            struct io_uring_sqe entry = {};
            entry.opcode = IORING_OP_ASYNC_CANCEL;
            entry.addr = operation_id(2, i, c->generation);
            entry.user_data = operation_id(6, i, c->generation);
            if (enqueue(x, &entry)) c->closing = 2;
        }
        int reclaim = (c->dead || c->eof) && !c->receiving &&
                      c->reply < 0 && c->closing != 1;
        if (reclaim) {
            __u32 queued;
            if (c->eof) {
                struct sockaddr_in *peer = (void *)(x->memory + bgs_ktcp_config.buffers +
                                           i * BGS_KTCP_RX_STRIDE + 4096);
                queued = load32(&x->page->llm_peer_pending[
                    bgs_pending_bucket(peer->sin_addr.s_addr, peer->sin_port)]);
            } else queued = load32(&x->page->llm_queued);
            /* Preserve corrected pending-acquire before fresh admission head. */
            reclaim = (s->stopping || !queued) &&
                      s->seen == load32(&x->page->llm_head);
        }
        if (reclaim) {
            struct io_uring_sqe entry = {};
            entry.opcode = IORING_OP_CLOSE; entry.fd = c->fd;
            entry.user_data = operation_id(4, i, c->generation);
            if (enqueue(x, &entry)) c->closing = 1;
        } else if (!c->dead && !c->eof && !c->receiving && !s->stopping) {
            struct io_uring_sqe entry = {};
            entry.opcode = IORING_OP_RECV; entry.fd = c->fd;
            entry.addr = bgs_ktcp_config.user_base + bgs_ktcp_config.buffers + i*BGS_KTCP_RX_STRIDE;
            entry.len = 4096; entry.user_data = operation_id(2, i, c->generation);
            if (enqueue(x, &entry)) c->receiving = 1;
        }
        context->previous = i; context->current = c->next;
    
    return 0;
}
static long connection_callback(__u64 index, void *opaque)
{
    struct connection_context *context=opaque;
    int result=progress_connection(context);
    if(result<0)context->failed=1;
    return result!=0;
}
static __noinline int progress_connections(struct step *x)
{
    struct bgs_ktcp_shared *s=x->s;
    struct connection_context context={.x=x,.previous=-1,.current=s->active};
    long result=bpf_loop(bgs_ktcp_config.capacity,connection_callback,&context,0);
    if(result<0||context.failed)return -1;
if (!s->stopping && s->accepting < 0 && s->free >= 0) {
        __u64 i = (__u32)s->free;
        asm volatile("%0 <<= 32; %0 >>= 32" : "+r"(i));
        if (i >= bgs_ktcp_config.capacity) return -1;
        struct bgs_ktcp_conn *c = connection(x, i);
        if (!c) return -1;
        int next = c->next;
        __u32 *length = (void *)(x->memory + bgs_ktcp_config.buffers +
                                i*BGS_KTCP_RX_STRIDE + 4112);
        *length = sizeof(struct sockaddr_in);
        struct io_uring_sqe entry = {};
        entry.opcode = IORING_OP_ACCEPT; entry.fd = bgs_ktcp_config.listener;
        entry.addr = bgs_ktcp_config.user_base + bgs_ktcp_config.buffers + i*BGS_KTCP_RX_STRIDE + 4096;
        entry.off = bgs_ktcp_config.user_base + bgs_ktcp_config.buffers + i*BGS_KTCP_RX_STRIDE + 4112;
        entry.user_data = operation_id(1, i, c->generation+1);
        if (enqueue(x, &entry)) {
            s->free = next; s->accepting = i;
            __u32 generation = c->generation+1;
            *c = (struct bgs_ktcp_conn){
                .fd=-1, .next=-1, .reply=-1, .last=-1, .generation=generation
            };
        }
    }
    return 0;
}
static __noinline int progress_timer(struct step *x)
{
    struct bgs_ktcp_shared *s = x->s;
    if (s->stopping && s->cancel_done &&
        s->timeout_active && !s->timeout_removing) {
        struct io_uring_sqe entry = {};
        entry.opcode = IORING_OP_TIMEOUT_REMOVE;
        entry.addr = 8ULL << 56; entry.user_data = operation_id(9, 0, 0);
        if (enqueue(x, &entry)) s->timeout_removing = 1;
    }
    __u64 now = s->loop.now_ns, deadline = s->loop.deadline_ns;
    int idle = s->seen == s->next_final &&
               s->seen == load32(&x->page->llm_head) &&
               !load32(&x->page->llm_queued) && now && deadline > now;
    if (!s->stopping && s->timeout_active && !s->timeout_updating &&
        (__u32)idle != s->timeout_idle) {
        struct __kernel_timespec *tick = (void *)(x->memory + bgs_ktcp_config.tick);
        __u64 ns = idle ? deadline-now : 60000;
        tick->tv_sec = ns / 1000000000ULL;
        tick->tv_nsec = ns % 1000000000ULL;
        struct io_uring_sqe entry = {};
        entry.opcode = IORING_OP_TIMEOUT_REMOVE;
        entry.addr = 8ULL << 56;
        entry.off = bgs_ktcp_config.user_base + bgs_ktcp_config.tick;
        entry.timeout_flags = KTCP_TIMEOUT_UPDATE;
        entry.user_data = operation_id(10, idle, 0);
        if (enqueue(x, &entry)) s->timeout_updating = 1;
    }
    return 0;
}
static __always_inline int submit(struct step *x, struct iou_ctx *loop_ctx)
{
    if (x->st == x->sh) return 0;
    publish(x);
    return bpf_io_uring_submit_sqes(loop_ctx, x->st-x->sh) < 0 ? -1 : 0;
}
SEC("struct_ops.s/kernel_tcp_step")
int BPF_PROG(kernel_tcp_step, struct iou_ctx *loop_ctx, struct iou_loop_params *lp)
{
    __u8 *memory = bpf_io_uring_get_region(loop_ctx, IOU_REGION_MEM, bgs_ktcp_config.mem_bytes);
    __u8 *rings = bpf_io_uring_get_region(loop_ctx, IOU_REGION_CQ, bgs_ktcp_config.ring_bytes);
    __u8 *sq = bpf_io_uring_get_region(loop_ctx, IOU_REGION_SQ,
                                      bgs_ktcp_config.entries*sizeof(struct io_uring_sqe));
    if (!memory || !rings || !sq) return IOU_LOOP_STOP;
    struct bgs_ktcp_shared *s = (void *)memory;
    if (s->loop.finished) return IOU_LOOP_STOP;
    __u64 now = bpf_ktime_get_ns(); s->loop.now_ns = now;
    if (!s->loop.model_quiesced && (s->loop.stop_requested ||
        (s->loop.deadline_ns && now >= s->loop.deadline_ns)))
        return IOU_LOOP_STOP;
    __u32 zero = 0;
    struct bgs_page *page = bpf_map_lookup_elem(&ctl, &zero);
    if (!page) return IOU_LOOP_STOP;
    if (s->loop.model_quiesced || *(volatile __u64 *)&page->stop_ns)
        s->stopping = 1;
    __u32 queued = load32(&page->llm_queued);
    if (queued > s->loop.backlog_peak) s->loop.backlog_peak = queued;
    struct step x = {
        .memory=memory, .rings=rings, .sq=sq, .s=s, .page=page,
        .sh=load32((void *)(rings+bgs_ktcp_config.sh)),
        .st=load32((void *)(rings+bgs_ktcp_config.st)),
        .cqhead=load32((void *)(rings+bgs_ktcp_config.ch))
    };
    if (drain_completions(&x)) { s->error=1; return IOU_LOOP_STOP; }
    if (!s->stopping && admit(page)) { s->error=2; return IOU_LOOP_STOP; }
    if (s->stopping && !s->cancelled) {
        struct io_uring_sqe entry = {};
        entry.opcode=IORING_OP_ASYNC_CANCEL;
        entry.cancel_flags=KTCP_CANCEL_ALL|KTCP_CANCEL_ANY;
        entry.user_data=operation_id(5,0,0);
        if (enqueue(&x,&entry)) s->cancelled=1;
    }
    if (s->stopping && !s->cancel_done) {
        if (submit(&x,loop_ctx)) {s->error=3;return IOU_LOOP_STOP;}
        lp->cq_wait_idx=x.cqhead+1;
        return IOU_LOOP_CONTINUE;
    }
    if (claim_requests(&x)) {s->error=4;return IOU_LOOP_STOP;}
    int active_requests=send_replies(&x);
    if (active_requests<0) {s->error=5;return IOU_LOOP_STOP;}
    if (progress_connections(&x)) {s->error=6;return IOU_LOOP_STOP;}
    if (progress_timer(&x)) {s->error=7;return IOU_LOOP_STOP;}
    s->loop.admission_seen=s->seen;
    s->loop.cq_seen=x.cqhead;
    if (submit(&x,loop_ctx)) {s->error=8;return IOU_LOOP_STOP;}
    if (s->stopping && s->cancelled && !s->pending &&
        s->active<0 && !active_requests) {
        s->head=load32(&page->llm_head);s->reserved=load32(&page->llm_reserved);
        s->sqhead=load32((void *)(rings+bgs_ktcp_config.sh));s->sqtail=x.st;
        s->cqhead=x.cqhead;s->cqtail=load32((void *)(rings+bgs_ktcp_config.ct));
        s->dropped=load32((void *)(rings+bgs_ktcp_config.dropped));
        s->overflow=load32((void *)(rings+bgs_ktcp_config.overflow));
        order_memory();s->loop.finished=1;
        return IOU_LOOP_STOP;
    }
    /* CQ consumer is CPU-owned here; newly completed inline submissions must
     * cause another iteration, while future GPU outputs use the existing timer. */
    lp->cq_wait_idx=x.cqhead+1;
    return IOU_LOOP_CONTINUE;
}
SEC(".struct_ops.link")
struct io_uring_bpf_ops io_loop = {
    .loop_step=(void *)kernel_tcp_step,
    .ring_fd=-1,
};
char _license[] SEC("license")="GPL";
