// SPDX-License-Identifier: Apache-2.0
// GPU-owned asynchronous TCP control around the unchanged eBPF/model queue.
#ifndef BPF_GPU_SERVER_IO_URING_TCP_H
#define BPF_GPU_SERVER_IO_URING_TCP_H
#include <cuda_runtime.h>
#include <utility>
#include "bpf_gpu_server_queue.h"
#include <linux/io_uring.h>
#include <bpf/bpf.h>
#ifdef BGS_URING_BPF_LOOP
#include <bpf/libbpf.h>
#include <bpf/btf.h>
#include <time.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
using UringIdentity=bgs_uring_context;
struct IoRing {
    unsigned *head,*tail,*cqhead,*cqtail,*flags,*dropped,*overflow;
    io_uring_sqe *sqes;io_uring_cqe *cqes;unsigned mask,cqmask;
#ifdef BGS_IO_COALESCED
    unsigned sq_head,sq_tail,sq_published;
#endif
};
struct IoConn { int fd,next,reply,last;unsigned receiving,closing,dead,eof,generation; };
struct IoRequest {
    int conn,next;unsigned seq,ng,header,sent,busy,active,final,header_ready;
    unsigned long long tag,client_ns;unsigned sends,early,bytes;
};
struct IoRecord {unsigned seq,gen,sends,early,bytes,aborted;unsigned long long client_ns;};
struct IoStats {
    bgs_uring_loop_control loop;
    unsigned accepted,closed,received,sends,bytes,completed,aborted,max_connections,max_requests;
    unsigned pending,error,head,reserved,dropped,overflow,sqhead,sqtail,cqhead,cqtail;
    unsigned trace_count,trace_overflow;IoRecord records[128];
};
__device__ unsigned io_load(const unsigned *p) {
    unsigned v;asm volatile("ld.acquire.sys.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ unsigned long long io_load64(const unsigned long long *p) {
    unsigned long long v;asm volatile("ld.acquire.sys.global.u64 %0,[%1];":"=l"(v):"l"(p):"memory");return v;
}
__device__ void io_store(unsigned *p,unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");
}
#ifdef BGS_IO_COALESCED
__device__ bool io_submit(IoRing &ring,io_uring_sqe entry,IoStats *stats) {
    if(ring.sq_tail-ring.sq_head>ring.mask)return false;
    ring.sqes[ring.sq_tail++&ring.mask]=entry;stats->pending++;return true;
}
__device__ void io_publish(IoRing &ring) {
    if(ring.sq_tail==ring.sq_published)return;
    __threadfence_system();io_store(ring.tail,ring.sq_tail);
    ring.sq_published=ring.sq_tail;
}
#else
__device__ bool io_submit(IoRing ring,io_uring_sqe entry,IoStats *stats) {
    unsigned tail=io_load(ring.tail);
    if(tail-io_load(ring.head)>ring.mask)return false;
    ring.sqes[tail&ring.mask]=entry;__threadfence_system();io_store(ring.tail,tail+1);stats->pending++;
    return true;
}
#endif
__device__ unsigned long long io_id(unsigned kind,unsigned value,unsigned generation=0) {
    return ((unsigned long long)kind<<56)|((unsigned long long)(generation&0xffffff)<<32)|value;
}
__device__ void io_retire(IoRequest *requests,IoConn *conns,unsigned i,IoStats *out,bool aborted) {
    auto &r=requests[i];
    if(r.conn>=0) {
        auto &c=conns[r.conn];c.reply=r.next;if(c.reply<0)c.last=-1;
    }
    unsigned index=out->trace_count++;
    if(index<128)out->records[index]={r.seq,r.ng,r.sends,r.early,r.bytes,(unsigned)aborted,r.client_ns};
    else out->trace_overflow++;
    out->completed++;if(aborted)out->aborted++;
    r.active=0;
}
struct IoState {
    IoRing ring;bgs_page *page;IoConn *conns;unsigned capacity;
    unsigned char *buffer;unsigned long long host_buffer;IoStats *out;int listener;
    int free=0,active=-1,accepting=-1;unsigned connections=0,seen=0,next_final=0;
    bool stopping=false,cancelled=false,cancel_done=false;
#ifdef BGS_URING_BPF_LOOP
    bool timeout_active=true,timeout_removing=false;
#ifndef BGS_IO_CQ_AFTER_SUBMIT
    bool timeout_idle=false,timeout_updating=false;
#endif
#endif
    IoRequest requests[BGS_LLM_SLOTS]={};
};
#ifdef BGS_IO_COALESCED
__device__ IoState *io_state;
#endif
__global__ void io_initialize(IoState *s) {
    if(threadIdx.x||blockIdx.x)return;
    for(unsigned i=0;i<s->capacity;i++)s->conns[i]={-1,i+1<s->capacity?(int)i+1:-1,-1,-1,0,0,0,0,0};
    for(auto &r:s->requests){r.conn=-1;r.next=-1;}
}
#if defined(BGS_URING_BPF_LOOP) && !defined(BGS_IO_CQ_AFTER_SUBMIT)
__device__ void io_acknowledge(IoStats *out,unsigned head) {
    if(io_load(&out->loop.cq_seen)!=head) {
        __threadfence_system();io_store(&out->loop.cq_seen,head);
    }
}
#endif
__device__ bool io_progress(IoState *s) {
    auto ring=s->ring;
#ifdef BGS_IO_COALESCED
    ring.sq_head=io_load(ring.head);ring.sq_tail=ring.sq_published=io_load(ring.tail);
#endif
    auto *page=s->page;auto *conns=s->conns;unsigned capacity=s->capacity;
    auto *buffer=s->buffer;auto host_buffer=s->host_buffer;auto *out=s->out;int listener=s->listener;
    auto &free=s->free;auto &active=s->active;auto &accepting=s->accepting;
    auto &connections=s->connections;auto &seen=s->seen;auto &next_final=s->next_final;
    auto &stopping=s->stopping;auto &cancelled=s->cancelled;auto &cancel_done=s->cancel_done;
#ifdef BGS_URING_BPF_LOOP
    auto &timeout_active=s->timeout_active;auto &timeout_removing=s->timeout_removing;
#endif
    auto *requests=s->requests;constexpr unsigned stride=4160,txstride=512;
        if(io_load64((unsigned long long*)&page->stop_ns))stopping=true;
        // Drain completions before publishing new work or reusing any buffer.
        unsigned ch=io_load(ring.cqhead);
#ifdef BGS_IO_COALESCED
        unsigned begin=ch,end=io_load(ring.cqtail);
        while(ch!=end) {
            auto e=ring.cqes[ch++&ring.cqmask];
#else
        while(ch!=io_load(ring.cqtail)) {
            auto e=ring.cqes[ch&ring.cqmask];io_store(ring.cqhead,++ch);
#endif
            unsigned kind=e.user_data>>56,index=(unsigned)e.user_data;
#ifdef BGS_URING_BPF_LOOP
            // A multishot timeout owns one pending operation until its final CQE.
            if(kind==8 && (e.flags&IORING_CQE_F_MORE))continue;
#endif
            out->pending--;
            if((e.user_data>>48)==BGS_IO_SEND_TAG || (e.user_data>>48)==BGS_IO_ABORT_TAG)kind=7;
            if(kind==1) {
                accepting=-1;auto &c=conns[index];
                if(e.res>=0) {
                    c.fd=e.res;c.next=active;active=index;connections++;out->accepted++;
                    if(connections>out->max_connections)out->max_connections=connections;
                    if(stopping)c.dead=1;
                }else {c.next=free;free=index;if(e.res!=-ECANCELED)out->error=1;}
            }else if(kind==2) {
                auto &c=conns[index];c.receiving=0;
                if(e.res>0)out->received++;
                else if(e.res==0)c.eof=1;
                // A transient socket read is not peer EOF or connection death.
                else if(e.res!=-EAGAIN && e.res!=-EINTR)c.dead=1;
            }else if(kind==3 || kind==7) {
                unsigned i=index%BGS_LLM_SLOTS;auto &r=requests[i];
                if(!r.active || !r.busy || r.tag!=e.user_data){out->error=2;continue;}
                r.busy=0;
                if((e.user_data>>48)==BGS_IO_ABORT_TAG) {
                    if(e.res!=0||r.seq!=next_final)out->error=3;else {io_retire(requests,conns,i,out,true);next_final++;}
                }else if(e.res>0) {
                    r.sent+=e.res;r.bytes+=e.res;out->bytes+=e.res;r.sends++;out->sends++;
                    if(!r.final)r.early++;
                    unsigned expected=(e.user_data>>32)&0xffff;
                    if(kind==7 && (unsigned)e.res==expected) {
                        if(r.seq!=next_final)out->error=3;
                        else {io_retire(requests,conns,i,out,false);next_final++;}
                    }
                }else {if(r.conn>=0)conns[r.conn].dead=1;}
            }else if(kind==4) {
                auto &c=conns[index];c.fd=-1;c.closing=0;out->closed++;
                // Removing from the active list occurs below after CQ drain.
            }else if(kind==5 || kind==6) {
                if(kind==5)cancel_done=true;
                // Cancellation completion; the target operation has its own CQE.
#ifdef BGS_URING_BPF_LOOP
            }else if(kind==8) {
                timeout_active=false;
                if(e.res!=-ECANCELED)out->error=5;
            }else if(kind==9) {
                // A timeout may be in deferred task work when cancel-all runs.
                if(e.res==-ENOENT)timeout_removing=false;
                else if(e.res<0)out->error=6;
#ifndef BGS_IO_CQ_AFTER_SUBMIT
            }else if(kind==10) {
                s->timeout_updating=false;
                if(e.res==0)s->timeout_idle=index;
                io_store(&out->loop.timer_may_idle,s->timeout_idle);
                if(e.res<0 && e.res!=-ENOENT && e.res!=-EALREADY && !(stopping && e.res==-ECANCELED))out->error=7;
#endif
#endif
            }else out->error=4;
        }
#ifdef BGS_IO_COALESCED
        if(ch!=begin)io_store(ring.cqhead,ch);
#endif
        if(stopping&&!cancelled) {
            io_uring_sqe e{};e.opcode=IORING_OP_ASYNC_CANCEL;e.cancel_flags=IORING_ASYNC_CANCEL_ALL|IORING_ASYNC_CANCEL_ANY;
            e.user_data=io_id(5,0);
            if(io_submit(ring,e,out))cancelled=true;
        }
        if(stopping&&!cancel_done) {
#ifdef BGS_IO_COALESCED
            io_publish(ring);
#endif
#if defined(BGS_URING_BPF_LOOP) && !defined(BGS_IO_CQ_AFTER_SUBMIT)
            io_acknowledge(out,ch);
#endif
            __nanosleep(1000);return true;
        }
#ifdef BGS_URING_BPF_LOOP
        if(stopping && cancel_done && timeout_active && !timeout_removing) {
            io_uring_sqe e{};e.opcode=IORING_OP_TIMEOUT_REMOVE;
            e.addr=8ULL<<56;e.user_data=io_id(9,0);
            if(io_submit(ring,e,out))timeout_removing=true;
        }
#endif

        // Claim eBPF slots in original admission order. Match by actual peer.
        while(seen!=io_load(&page->llm_head)) {
            unsigned i=seen%BGS_LLM_SLOTS;auto &r=requests[i];auto *slot=&page->llm[i];
            unsigned state=io_load(&slot->state);
            if(r.active||(state!=BGS_PENDING&&state!=BGS_DONE))break;
            int peer=active;
            while(peer>=0) {
                auto *addr=(sockaddr_in*)(buffer+(unsigned)peer*stride+4096);
                if(slot->addr_be==addr->sin_addr.s_addr&&slot->port_be==addr->sin_port)break;
                peer=conns[peer].next;
            }
            if(peer<0&&!stopping)break;
            r={peer,-1,seen,io_load(&slot->n_gen),0,0,0,1,0,0,0,io_load64(&slot->client_ns),0,0,0};
            if(peer>=0) {
                auto &c=conns[peer];
                if(c.last>=0)requests[c.last].next=i;else c.reply=i;
                c.last=i;
            }
            seen++;
        }
        unsigned active_requests=0;
        for(unsigned i=0;i<BGS_LLM_SLOTS;++i) {
            auto &r=requests[i];if(!r.active)continue;active_requests++;
            if(r.busy)continue;
            auto *slot=&page->llm[i];unsigned state=io_load(&slot->state);
            bool gone=r.conn<0||conns[r.conn].dead||stopping;
            if(gone) {
                if(r.seq!=next_final)continue;
                if(state!=BGS_DONE) {
                    if(!stopping||!io_load(&out->loop.model_quiesced))continue;
                    __threadfence_system();io_store(&slot->state,BGS_DONE);
                }
                io_uring_sqe e{};e.opcode=IORING_OP_NOP;e.user_data=(BGS_IO_ABORT_TAG<<48)|r.seq;
                if(io_submit(ring,e,out)){r.busy=1;r.tag=e.user_data;r.final=1;}
                continue;
            }
            auto &c=conns[r.conn];if(c.reply!=(int)i)continue;
            unsigned short flags=__ldcg(&slot->pad);
            if(flags&BGS_LLM_VALIDATING)continue;
            bool bad=flags&BGS_LLM_REJECTED;
            if(bad && (state!=BGS_DONE || r.seq!=next_final))continue;
            if(bad && (flags&BGS_LLM_TRANSPORT_MASK)==1){c.dead=1;continue;}
            auto *tx=buffer+capacity*stride+i*txstride;
            if(!r.header_ready) {
                r.header_ready=1;
                if((flags&BGS_LLM_TRANSPORT_MASK)==2) {
                    const char ok[]="HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: ";
                    const char error[]="HTTP/1.1 400 Bad Request\r\nContent-Type: application/octet-stream\r\nContent-Length: ";
                    const char *prefix=bad?error:ok;
                    for(unsigned j=0;prefix[j];++j)tx[r.header++]=prefix[j];
                    unsigned value=bad?0:r.ng*4,n=0;char digits[4];
                    do{digits[n++]=value%10+'0';value/=10;}while(value);
                    while(n)tx[r.header++]=digits[--n];
                    const char end[]="\r\n\r\n";for(unsigned j=0;j<4;++j)tx[r.header++]=end[j];
                }
            }
            unsigned produced=bad?0:io_load(&slot->produced);
            // Publish final responses in admission order: physical-slot order
            // can expose a later completion while the wrap target is reserved.
            if((state!=BGS_DONE || r.seq!=next_final)&&produced==r.ng&&produced)produced--;
            bool final=state==BGS_DONE&&r.seq==next_final&&(bad||produced==r.ng);
            if(!produced&&!bad)continue;
            unsigned total=r.header+produced*4;if(total<=r.sent)continue;
#ifdef BGS_IO_FULL_COPY
            unsigned first=0; // matched control for the former prefix-copy loop
#else
            // Keep a partially sent token, but skip every fully sent token.
            unsigned first=r.sent>r.header?(r.sent-r.header)/4:0;
#endif
            for(unsigned j=first;j<produced;++j) {
                unsigned token=io_load(&slot->tok_out[j]);
                for(unsigned byte=0;byte<4;++byte)tx[r.header+j*4+byte]=(token>>(byte*8))&255;
            }
            io_uring_sqe e{};e.opcode=IORING_OP_SEND;e.fd=c.fd;e.addr=host_buffer+capacity*stride+i*txstride+r.sent;
            e.len=total-r.sent;e.msg_flags=MSG_NOSIGNAL;
            e.user_data=final?((BGS_IO_SEND_TAG<<48)|((unsigned long long)e.len<<32)|r.seq):io_id(3,r.seq);
            if(io_submit(ring,e,out)){r.busy=1;r.tag=e.user_data;r.final=final;}
        }
        if(active_requests>out->max_requests)out->max_requests=active_requests;
        // Rearm receive and close only after replies and receive references end.
        int *link=&active;
        while(*link>=0) {
            unsigned i=*link;auto &c=conns[i];
            if(c.fd<0&&!c.closing) {
                *link=c.next;c.next=free;free=i;connections--;continue;
            }
            if(stopping)c.dead=1;
            if(c.dead&&c.receiving&&!c.closing) {
                io_uring_sqe e{};e.opcode=IORING_OP_ASYNC_CANCEL;e.addr=io_id(2,i,c.generation);e.user_data=io_id(6,i,c.generation);
                if(io_submit(ring,e,out))c.closing=2; // wait for receive CQE
            }
            bool reclaim=(c.dead||c.eof)&&!c.receiving&&c.reply<0&&c.closing!=1;
            if(reclaim) {
                unsigned queued;
#ifndef BGS_IO_GLOBAL_PENDING_GUARD
                if(c.eof) {
                    // RECV EOF ends this socket's producer. Pending references
                    // protect frames not yet published into a model slot.
                    auto *peer=(sockaddr_in*)(buffer+i*stride+4096);
                    queued=io_load(&page->llm_peer_pending[
                        bgs_pending_bucket(peer->sin_addr.s_addr,peer->sin_port)]);
                }else
#endif
                    queued=io_load(&page->llm_queued);
                // Acquire pending first, then freshly observe admission/head.
                // A consumer publishes the slot/head before releasing pending.
                reclaim=(stopping||!queued)&&seen==io_load(&page->llm_head);
            }
            if(reclaim) {
                io_uring_sqe e{};e.opcode=IORING_OP_CLOSE;e.fd=c.fd;e.user_data=io_id(4,i,c.generation);
                if(io_submit(ring,e,out))c.closing=1;
            }else if(!c.dead&&!c.eof&&!c.receiving&&!stopping) {
                io_uring_sqe e{};e.opcode=IORING_OP_RECV;e.fd=c.fd;e.addr=host_buffer+i*stride;e.len=4096;e.user_data=io_id(2,i,c.generation);
                if(io_submit(ring,e,out))c.receiving=1;
            }
            link=&c.next;
        }
        if(!stopping&&accepting<0&&free>=0) {
            unsigned i=free;auto &c=conns[i];int next=c.next;
            auto *len=(socklen_t*)(buffer+i*stride+4112);*len=sizeof(sockaddr_in);
            io_uring_sqe e{};e.opcode=IORING_OP_ACCEPT;e.fd=listener;e.addr=host_buffer+i*stride+4096;
            e.off=host_buffer+i*stride+4112;e.user_data=io_id(1,i,c.generation+1);
            if(io_submit(ring,e,out)){free=next;accepting=i;c={-1,-1,-1,-1,0,0,0,0,c.generation+1};}
        }
#if defined(BGS_URING_BPF_LOOP) && !defined(BGS_IO_CQ_AFTER_SUBMIT)
        unsigned long long deadline=io_load64(&out->loop.deadline_ns),now=io_load64(&out->loop.now_ns);
        bool idle=seen==next_final && seen==io_load(&page->llm_head) &&
                  !io_load(&page->llm_queued) && now && deadline>now;
        if(!stopping && timeout_active && !s->timeout_updating && idle!=s->timeout_idle) {
            auto *tick=(__kernel_timespec*)(buffer+capacity*stride+BGS_LLM_SLOTS*txstride);
            unsigned long long ns=idle ? deadline-now : 60000;
            tick->tv_sec=ns/1000000000ULL;tick->tv_nsec=ns%1000000000ULL;
            io_uring_sqe e{};e.opcode=IORING_OP_TIMEOUT_REMOVE;e.addr=8ULL<<56;
            e.off=host_buffer+capacity*stride+BGS_LLM_SLOTS*txstride;
            e.timeout_flags=IORING_TIMEOUT_UPDATE;e.user_data=io_id(10,idle);
            if(idle)io_store(&out->loop.timer_may_idle,1);
            if(io_submit(ring,e,out))s->timeout_updating=true;
            else io_store(&out->loop.timer_may_idle,s->timeout_idle);
        }
#endif
#ifdef BGS_IO_COALESCED
        io_publish(ring);
#endif
#if defined(BGS_URING_BPF_LOOP) && !defined(BGS_IO_CQ_AFTER_SUBMIT)
        if(io_load(&out->loop.admission_seen)!=seen)io_store(&out->loop.admission_seen,seen);
        io_acknowledge(out,ch);
#endif
        if(stopping&&cancelled&&!out->pending&&active<0&&!active_requests) {
    out->head=io_load(&page->llm_head);out->reserved=io_load(&page->llm_reserved);
    out->sqhead=io_load(ring.head);out->sqtail=io_load(ring.tail);
    out->cqhead=io_load(ring.cqhead);out->cqtail=io_load(ring.cqtail);
    out->dropped=io_load(ring.dropped);out->overflow=io_load(ring.overflow);
    __threadfence_system();
#ifdef BGS_URING_BPF_LOOP
    io_store(&out->loop.finished,1);
#endif
        return false;
        }

    __nanosleep(1000);return true;
}
__global__ void io_network(IoState *s) {
    if(threadIdx.x||blockIdx.x)return;
    while(io_progress(s)){}
}
static void io_check(cudaError_t status,const char *what) {
    if(status){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(status));exit(10);}
}
static size_t io_align(size_t bytes){return (bytes+4095)&~(size_t)4095;}
static void *io_memory(size_t bytes) {
    void *p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED){perror("io mmap");exit(11);}
    memset(p,0,bytes);io_check(cudaHostRegister(p,bytes,cudaHostRegisterMapped),"io register");return p;
}
static void *io_device(void *p) {
    void *d;io_check(cudaHostGetDevicePointer(&d,p,0),"io device pointer");return d;
}
struct UringNetwork {
    void *sq,*rings,*buffer;IoStats *report;IoConn *conns;IoState *state;int ringfd;
    size_t sq_bytes,ring_bytes,buffer_bytes;cudaStream_t stream;
#ifdef BGS_URING_BPF_LOOP
    bpf_object *loop_object;bpf_link *loop_link;
#endif
};
#ifdef BGS_URING_BPF_LOOP
static bgs_uring_loop_control *io_signal_control=nullptr;
#ifndef BGS_IO_CQ_AFTER_SUBMIT
static volatile sig_atomic_t io_loop_tid=0;
#endif
static void uring_interrupt() {
    if(io_signal_control) {
        __atomic_store_n(&io_signal_control->stop_requested,1,__ATOMIC_RELEASE);
#ifndef BGS_IO_CQ_AFTER_SUBMIT
        // A signal on another thread must also wake the sleeping loop owner.
        sig_atomic_t tid=io_loop_tid;
        if(tid && syscall(__NR_gettid)!=tid)syscall(__NR_tgkill,getpid(),tid,SIGTERM);
#endif
    }
}
static int uring_attach_loop(UringNetwork *net) {
    io_uring_region_desc region{};
    region.user_addr=(uint64_t)net->report;region.size=65536;region.flags=IORING_MEM_REGION_TYPE_USER;
    io_uring_mem_region_reg reg{};reg.region_uptr=(uint64_t)&region;
    if(syscall(__NR_io_uring_register,net->ringfd,IORING_REGISTER_MEM_REGION,&reg,1)<0){perror("io loop memory");return 1;}
#ifdef BGS_IO_CQ_AFTER_SUBMIT
    net->loop_object=bpf_object__open_file(BGS_BUILD_DIR "/uring_loop_after.bpf.o",nullptr);
#else
    net->loop_object=bpf_object__open_file(BGS_BUILD_DIR "/uring_loop.bpf.o",nullptr);
#endif
    if(!net->loop_object || libbpf_get_error(net->loop_object)){fprintf(stderr,"io loop open failed\n");return 1;}
    const char *names[]={"ctl","pending"};
    const char *paths[]={"/sys/fs/bpf/bgs_ctl","/sys/fs/bpf/bgs_pending"};
    for(unsigned i=0;i<2;i++) {
        int fd=bpf_obj_get(paths[i]);bpf_map_info info{};unsigned bytes=sizeof(info);
        auto *shared=bpf_object__find_map_by_name(net->loop_object,names[i]);
        if(fd<0||!shared||bpf_obj_get_info_by_fd(fd,&info,&bytes)){if(fd>=0)close(fd);return 1;}
        bpf_map__set_max_entries(shared,info.max_entries);
        int error=bpf_map__reuse_fd(shared,fd);close(fd);if(error)return 1;
    }
    auto *map=bpf_object__find_map_by_name(net->loop_object,"io_loop");
    if(!map)return 1;
    size_t size;auto *value=(char*)bpf_map__initial_value(map,&size);
    auto *btf=bpf_object__btf(net->loop_object);
    int type_id=btf__find_by_name_kind(btf,"io_uring_bpf_ops",BTF_KIND_STRUCT);
    auto *type=type_id>0 ? btf__type_by_id(btf,type_id) : nullptr;
    bool assigned=false;
    if(type && value)for(unsigned i=0;i<btf_vlen(type);++i) {
        auto *member=btf_members(type)+i;
        if(!strcmp(btf__name_by_offset(btf,member->name_off),"ring_fd")) {
            unsigned offset=btf_member_bit_offset(type,i)/8;
            if(offset+sizeof(unsigned)>size)return 1;
            memcpy(value+offset,&net->ringfd,sizeof(unsigned));assigned=true;break;
        }
    }
    if(!assigned || bpf_object__load(net->loop_object)){fprintf(stderr,"io loop load failed\n");return 1;}
    net->loop_link=bpf_map__attach_struct_ops(map);
    if(!net->loop_link || libbpf_get_error(net->loop_link)){fprintf(stderr,"io loop attach failed\n");return 1;}
    io_signal_control=&net->report->loop;
    return 0;
}
static int uring_enter_loop(UringNetwork *net) {
    int result=syscall(__NR_io_uring_enter,net->ringfd,0,0,0,nullptr,0);
    if(result<0 && errno!=EINTR){perror("io kernel loop");return 1;}
    return 0;
}
static int uring_wait(UringNetwork *net,unsigned seconds,volatile sig_atomic_t *stopped) {
#ifndef BGS_IO_CQ_AFTER_SUBMIT
    io_loop_tid=syscall(__NR_gettid);
#endif
    timespec now{};clock_gettime(CLOCK_MONOTONIC,&now);
    unsigned long long deadline=(unsigned long long)now.tv_sec*1000000000+now.tv_nsec+(unsigned long long)seconds*1000000000;
    __atomic_store_n(&net->report->loop.deadline_ns,deadline,__ATOMIC_RELEASE);
    while(!*stopped) {
        if(uring_enter_loop(net))return 1;
        clock_gettime(CLOCK_MONOTONIC,&now);
        if((unsigned long long)now.tv_sec*1000000000+now.tv_nsec>=deadline)break;
        if(__atomic_load_n(&net->report->loop.stop_requested,__ATOMIC_ACQUIRE))break;
    }
    return 0;
}
static int uring_drain(UringNetwork *net) {
    while(!__atomic_load_n(&net->report->loop.finished,__ATOMIC_ACQUIRE))
        if(uring_enter_loop(net))return 1;
    return 0;
}
#endif
static int uring_start(bgs_page *page,bgs_page *device,int listener,UringNetwork *net) {
    int streams=bpf_obj_get("/sys/fs/bpf/bgs_streams");
    bpf_map_info info{};unsigned size=sizeof(info);
    if(streams<0||bpf_obj_get_info_by_fd(streams,&info,&size)){perror("io stream map");return 1;}
    close(streams);unsigned capacity=info.max_entries,entries=1;
    while(entries<capacity+BGS_LLM_SLOTS+2)entries*=2;
    net->sq_bytes=io_align(entries*sizeof(io_uring_sqe));
    net->ring_bytes=io_align(4096+entries*2*sizeof(io_uring_cqe));
    net->buffer_bytes=io_align((size_t)capacity*4160+BGS_LLM_SLOTS*512
#ifdef BGS_URING_BPF_LOOP
        +sizeof(__kernel_timespec)
#endif
    );
    net->sq=io_memory(net->sq_bytes);net->rings=io_memory(net->ring_bytes);net->buffer=io_memory(net->buffer_bytes);
    net->report=(IoStats*)io_memory(65536);
    io_check(cudaMalloc(&net->conns,capacity*sizeof(IoConn)),"io connections");
    auto *dsq=(char*)io_device(net->sq),*dring=(char*)io_device(net->rings);
    auto *dbuffer=(unsigned char*)io_device(net->buffer);auto *dreport=(IoStats*)io_device(net->report);
    int identity=bpf_obj_get("/sys/fs/bpf/bgs_uring_owner");unsigned key=0;
    struct stat namespace_info{};
    if(stat("/proc/self/ns/pid",&namespace_info)){perror("io PID namespace");return 2;}
    UringIdentity owner{(unsigned)getpid(),0,0,(unsigned long long)namespace_info.st_dev,(unsigned long long)namespace_info.st_ino};
    if(identity<0||bpf_map_update_elem(identity,&key,&owner,BPF_ANY)){perror("io identity");return 2;}
    io_uring_params p{};p.flags=IORING_SETUP_NO_MMAP|IORING_SETUP_NO_SQARRAY;
#ifdef BGS_URING_BPF_LOOP
    p.flags|=IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN;
#else
    p.flags|=IORING_SETUP_SQPOLL;
#endif
    p.sq_off.user_addr=(uint64_t)net->sq;p.cq_off.user_addr=(uint64_t)net->rings;
    net->ringfd=syscall(__NR_io_uring_setup,entries,&p);
    if(net->ringfd<0){perror("io setup");return 3;}
    if(p.cq_off.cqes+p.cq_entries*sizeof(io_uring_cqe)>net->ring_bytes)return 4;
    if(bpf_map_lookup_elem(identity,&key,&owner)){perror("io bind lookup");return 5;}
    if(!owner.ring){fprintf(stderr,"io PID namespace binding returned no ring\n");return 5;}
#ifdef BGS_URING_BPF_LOOP
    owner.pad=1;
    if(bpf_map_update_elem(identity,&key,&owner,BPF_ANY)){perror("io queue admission");close(identity);return 5;}
#endif
    close(identity);
    IoRing ring{};
    ring.head=(unsigned*)(dring+p.sq_off.head);ring.tail=(unsigned*)(dring+p.sq_off.tail);
    ring.cqhead=(unsigned*)(dring+p.cq_off.head);ring.cqtail=(unsigned*)(dring+p.cq_off.tail);
    ring.flags=(unsigned*)(dring+p.sq_off.flags);ring.dropped=(unsigned*)(dring+p.sq_off.dropped);
    ring.overflow=(unsigned*)(dring+p.cq_off.overflow);ring.sqes=(io_uring_sqe*)dsq;ring.cqes=(io_uring_cqe*)(dring+p.cq_off.cqes);
    ring.mask=*(unsigned*)((char*)net->rings+p.sq_off.ring_mask);
    ring.cqmask=*(unsigned*)((char*)net->rings+p.cq_off.ring_mask);
    // A bootstrap NOP starts an observed active SQPOLL epoch.
    auto *sqe=(io_uring_sqe*)net->sq;sqe[0].opcode=IORING_OP_NOP;sqe[0].user_data=0xb007;
    __atomic_store_n((unsigned*)((char*)net->rings+p.sq_off.tail),1,__ATOMIC_RELEASE);
#ifdef BGS_URING_BPF_LOOP
    if(syscall(__NR_io_uring_enter,net->ringfd,1,1,IORING_ENTER_GETEVENTS,nullptr,0)<0)return 6;
#else
    if(syscall(__NR_io_uring_enter,net->ringfd,0,0,IORING_ENTER_SQ_WAKEUP,nullptr,0)<0)return 6;
#endif
    while(!__atomic_load_n((unsigned*)((char*)net->rings+p.cq_off.tail),__ATOMIC_ACQUIRE))usleep(1000);
    auto *cqe=(io_uring_cqe*)((char*)net->rings+p.cq_off.cqes);
    if(cqe[0].user_data!=0xb007||cqe[0].res)return 7;
    __atomic_store_n((unsigned*)((char*)net->rings+p.cq_off.head),1,__ATOMIC_RELEASE);
#ifdef BGS_URING_BPF_LOOP
    net->report->loop.sq_head=p.sq_off.head;net->report->loop.sq_tail=p.sq_off.tail;net->report->loop.cq_tail=p.cq_off.tail;
    auto *tick=(__kernel_timespec*)((char*)net->buffer+capacity*4160+BGS_LLM_SLOTS*512);
    tick->tv_sec=0;tick->tv_nsec=60000;
    sqe[1]={};sqe[1].opcode=IORING_OP_TIMEOUT;sqe[1].addr=(uint64_t)tick;sqe[1].len=1;
    sqe[1].timeout_flags=IORING_TIMEOUT_MULTISHOT;sqe[1].user_data=8ULL<<56;
    net->report->pending=1;
    __atomic_store_n((unsigned*)((char*)net->rings+p.sq_off.tail),2,__ATOMIC_RELEASE);
    if(syscall(__NR_io_uring_enter,net->ringfd,1,0,0,nullptr,0)<0)return 8;
    if(uring_attach_loop(net))return 9;
#endif
    io_check(cudaStreamCreateWithFlags(&net->stream,cudaStreamNonBlocking),"io stream");
    IoState initial{};
    initial.ring=ring;initial.page=device;initial.conns=net->conns;initial.capacity=capacity;
    initial.buffer=dbuffer;initial.host_buffer=(uint64_t)net->buffer;initial.out=dreport;initial.listener=listener;
    io_check(cudaMalloc(&net->state,sizeof(IoState)),"io state");
    io_check(cudaMemcpy(net->state,&initial,sizeof(initial),cudaMemcpyHostToDevice),"io state upload");
    io_initialize<<<1,1>>>(net->state);io_check(cudaGetLastError(),"io initialize");
    io_check(cudaStreamSynchronize(0),"io initialized");
#ifdef BGS_IO_COALESCED
    io_check(cudaMemcpyToSymbol(io_state,&net->state,sizeof(net->state)),"io dispatcher state");
    puts("network=coalesced GPU dispatcher; shutdown-only network worker");
#else
    io_network<<<1,1,0,net->stream>>>(net->state);
    io_check(cudaGetLastError(),"io launch");
#endif
    printf("NET_READY pid=%d port=%u ring_bound=1 capacity=%u sq=%u cq=%u\n",getpid(),BGS_LLM_TCP_PORT,capacity,p.sq_entries,p.cq_entries);fflush(stdout);
    return 0;
}
static void uring_quiesce(UringNetwork *net) {
    __atomic_store_n(&net->report->loop.model_quiesced,1,__ATOMIC_RELEASE);
#ifdef BGS_IO_COALESCED
    // Model root and all children have quiesced; the host kernel loop drains
    // cancellation/CQ work while this shutdown-only worker retains I/O state.
    io_network<<<1,1,0,net->stream>>>(net->state);
    io_check(cudaGetLastError(),"io shutdown launch");
#endif
#ifndef BGS_URING_BPF_LOOP
    syscall(__NR_io_uring_enter,net->ringfd,0,0,IORING_ENTER_SQ_WAKEUP,nullptr,0);
#endif
}
static int uring_finish(UringNetwork *net) {
    io_check(cudaStreamSynchronize(net->stream),"io finish");
    auto &r=*net->report;
#ifdef BGS_URING_BPF_LOOP
    printf("ADMISSION queued_peak=%u\n",r.loop.backlog_peak);
#endif
    printf("RESULT {\"accepted\":%u,\"closed\":%u,\"received\":%u,\"sends\":%u,\"bytes\":%u,"
           "\"completed\":%u,\"aborted\":%u,\"max_connections\":%u,\"max_requests\":%u,\"pending\":%u,"
           "\"error\":%u,\"head\":%u,\"reserved\":%u,\"dropped\":%u,\"overflow\":%u,"
           "\"sqhead\":%u,\"sqtail\":%u,\"cqhead\":%u,\"cqtail\":%u,\"trace_count\":%u,\"trace_overflow\":%u}\n",
           r.accepted,r.closed,r.received,r.sends,r.bytes,r.completed,r.aborted,r.max_connections,r.max_requests,
           r.pending,r.error,r.head,r.reserved,r.dropped,r.overflow,r.sqhead,r.sqtail,r.cqhead,r.cqtail,r.trace_count,r.trace_overflow);
    for(unsigned i=0;i<r.trace_count&&i<128;++i) {
        auto &v=r.records[i];
        printf("REPLY {\"seq\":%u,\"gen\":%u,\"sends\":%u,\"early\":%u,\"bytes\":%u,\"aborted\":%u,\"client_ns\":%llu}\n",
               v.seq,v.gen,v.sends,v.early,v.bytes,v.aborted,v.client_ns);
    }
    unsigned error=r.error;
#ifdef BGS_URING_BPF_LOOP
    io_signal_control=nullptr;
#ifndef BGS_IO_CQ_AFTER_SUBMIT
    io_loop_tid=0;
#endif
    bpf_link__destroy(net->loop_link);bpf_object__close(net->loop_object);
#endif
    close(net->ringfd);io_check(cudaFree(net->conns),"io connections free");
    io_check(cudaFree(net->state),"io state free");
    for(auto pair:{std::pair<void*,size_t>{net->sq,net->sq_bytes},{net->rings,net->ring_bytes},{net->buffer,net->buffer_bytes},{net->report,65536}}) {
        io_check(cudaHostUnregister(pair.first),"io unregister");munmap(pair.first,pair.second);
    }
    io_check(cudaStreamDestroy(net->stream),"io stream destroy");
    puts("CLEANUP ring_connections_buffers_released");fflush(stdout);
    return error?1:0;
}

#endif
