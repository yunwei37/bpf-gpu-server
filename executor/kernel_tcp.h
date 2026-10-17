// SPDX-License-Identifier: Apache-2.0
// Experimental BPF-owned TCP transport. No steady GPU I/O kernel.
#ifndef BGS_KERNEL_TCP_HOST_H
#define BGS_KERNEL_TCP_HOST_H
#include <cuda_runtime.h>
#include <linux/io_uring.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <bpf/btf.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include "kernel_tcp_state.h"

static_assert(IORING_CQE_F_MORE==(1u<<1),"BPF UAPI flag compatibility");
static_assert(IORING_ASYNC_CANCEL_ALL==(1u<<0),"BPF UAPI flag compatibility");
static_assert(IORING_ASYNC_CANCEL_ANY==(1u<<2),"BPF UAPI flag compatibility");
static_assert(IORING_TIMEOUT_UPDATE==(1u<<1),"BPF UAPI flag compatibility");
static_assert(MSG_NOSIGNAL==0x4000,"BPF socket flag compatibility");

struct UringNetwork {
    void *sq=nullptr,*rings=nullptr,*memory=nullptr;
    size_t sq_bytes=0,ring_bytes=0,mem_bytes=0;
    bgs_ktcp_layout layout{};
    bgs_ktcp_shared *report=nullptr;
    int ringfd=-1;
    bpf_object *loop_object=nullptr;
    bpf_link *loop_link=nullptr;
};
static bgs_uring_loop_control *io_signal_control=nullptr;
static volatile sig_atomic_t io_loop_tid=0;
static size_t io_align(size_t size){return (size+4095)&~size_t(4095);}
static void *kernel_memory(size_t size) {
    void *p=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED){perror("kernel TCP mmap");return nullptr;}
    memset(p,0,size);return p;
}
static void uring_interrupt() {
    if(!io_signal_control)return;
    __atomic_store_n(&io_signal_control->stop_requested,1,__ATOMIC_RELEASE);
    sig_atomic_t tid=io_loop_tid;
    if(tid && syscall(__NR_gettid)!=tid)
        syscall(__NR_tgkill,getpid(),tid,SIGTERM);
}
static int kernel_configure(bpf_object *object,const bgs_ktcp_layout &layout) {
    auto *btf=bpf_object__btf(object);
    int section_id=btf__find_by_name_kind(btf,".rodata",BTF_KIND_DATASEC);
    auto *section=section_id>0?btf__type_by_id(btf,section_id):nullptr;
    if(!section)return 1;
    auto *vars=btf_var_secinfos(section);
    for(unsigned i=0;i<btf_vlen(section);i++) {
        auto *var=btf__type_by_id(btf,vars[i].type);
        const char *name=var?btf__name_by_offset(btf,var->name_off):nullptr;
        if(!name || strcmp(name,"bgs_ktcp_config"))continue;
        bpf_map *map;
        bpf_object__for_each_map(map,object) {
            if(bpf_map__btf_value_type_id(map)!=(unsigned)section_id)continue;
            size_t size=0;void *initial=(void*)bpf_map__initial_value(map,&size);
            if(!initial || vars[i].size!=sizeof(layout) ||
               vars[i].offset+sizeof(layout)>size)return 1;
            memcpy((char*)initial+vars[i].offset,&layout,sizeof(layout));return 0;
        }
    }
    return 1;
}
static int kernel_attach(UringNetwork *net) {
    io_uring_region_desc region{};
    region.user_addr=(uint64_t)net->memory;region.size=net->mem_bytes;
    region.flags=IORING_MEM_REGION_TYPE_USER;
    io_uring_mem_region_reg registration{};
    registration.region_uptr=(uint64_t)&region;
    if(syscall(__NR_io_uring_register,net->ringfd,IORING_REGISTER_MEM_REGION,
               &registration,1)<0){perror("kernel TCP memory region");return 1;}
    net->loop_object=bpf_object__open_file(BGS_BUILD_DIR "/kernel_tcp.bpf.o",nullptr);
    if(!net->loop_object || libbpf_get_error(net->loop_object)) {
        fprintf(stderr,"kernel TCP BPF open failed\n");return 1;
    }
    const char *names[]={"ctl","pending"};
    const char *paths[]={"/sys/fs/bpf/bgs_ctl","/sys/fs/bpf/bgs_pending"};
    for(unsigned i=0;i<2;i++) {
        int fd=bpf_obj_get(paths[i]);bpf_map_info info{};unsigned bytes=sizeof(info);
        auto *map=bpf_object__find_map_by_name(net->loop_object,names[i]);
        if(fd<0||!map||bpf_obj_get_info_by_fd(fd,&info,&bytes)){
            if(fd>=0)close(fd);return 1;
        }
        bpf_map__set_max_entries(map,info.max_entries);
        int error=bpf_map__reuse_fd(map,fd);close(fd);
        if(error)return 1;
    }
    if(kernel_configure(net->loop_object,net->layout)){
        fprintf(stderr,"kernel TCP readonly layout initialization failed\n");return 1;
    }
    auto *map=bpf_object__find_map_by_name(net->loop_object,"io_loop");
    if(!map)return 1;
    size_t size=0;auto *value=(char*)bpf_map__initial_value(map,&size);
    auto *btf=bpf_object__btf(net->loop_object);
    int type_id=btf__find_by_name_kind(btf,"io_uring_bpf_ops",BTF_KIND_STRUCT);
    auto *type=type_id>0?btf__type_by_id(btf,type_id):nullptr;
    bool assigned=false;
    if(type&&value)for(unsigned i=0;i<btf_vlen(type);i++){
        auto *member=btf_members(type)+i;
        if(strcmp(btf__name_by_offset(btf,member->name_off),"ring_fd"))continue;
        unsigned offset=btf_member_bit_offset(type,i)/8;
        if(offset+sizeof(unsigned)>size)return 1;
        memcpy(value+offset,&net->ringfd,sizeof(unsigned));assigned=true;break;
    }
    if(!assigned||bpf_object__load(net->loop_object)){
        fprintf(stderr,"kernel TCP BPF load failed\n");return 1;
    }
    net->loop_link=bpf_map__attach_struct_ops(map);
    if(!net->loop_link || libbpf_get_error(net->loop_link)){
        fprintf(stderr,"kernel TCP BPF attach failed\n");return 1;
    }
    io_signal_control=&net->report->loop;
    return 0;
}
static int uring_enter_loop(UringNetwork *net) {
    int result=syscall(__NR_io_uring_enter,net->ringfd,0,0,0,nullptr,0);
    if(result<0 && errno!=EINTR){perror("kernel TCP enter");return 1;}
    if(__atomic_load_n(&net->report->error,__ATOMIC_ACQUIRE)){
        fprintf(stderr,"kernel TCP state error=%u\n",net->report->error);return 1;
    }
    return 0;
}
static int uring_wait(UringNetwork *net,unsigned seconds,volatile sig_atomic_t *stopped){
    io_loop_tid=syscall(__NR_gettid);
    timespec now{};clock_gettime(CLOCK_MONOTONIC,&now);
    unsigned long long deadline=(unsigned long long)now.tv_sec*1000000000ULL+
                                now.tv_nsec+(unsigned long long)seconds*1000000000ULL;
    __atomic_store_n(&net->report->loop.deadline_ns,deadline,__ATOMIC_RELEASE);
    while(!*stopped){
        if(uring_enter_loop(net))return 1;
        clock_gettime(CLOCK_MONOTONIC,&now);
        if((unsigned long long)now.tv_sec*1000000000ULL+now.tv_nsec>=deadline)break;
        if(__atomic_load_n(&net->report->loop.stop_requested,__ATOMIC_ACQUIRE))break;
    }
    return 0;
}
static void uring_quiesce(UringNetwork *net){
    __atomic_store_n(&net->report->loop.model_quiesced,1,__ATOMIC_RELEASE);
}
static int uring_drain(UringNetwork *net){
    while(!__atomic_load_n(&net->report->loop.finished,__ATOMIC_ACQUIRE))
        if(uring_enter_loop(net))return 1;
    return 0;
}
static int uring_start(bgs_page *page,bgs_page *device,int listener,UringNetwork *net){
    (void)page;(void)device;
    int streams=bpf_obj_get("/sys/fs/bpf/bgs_streams");
    bpf_map_info info{};unsigned size=sizeof(info);
    if(streams<0||bpf_obj_get_info_by_fd(streams,&info,&size)){
        if(streams>=0)close(streams);perror("kernel TCP stream map");return 1;
    }
    close(streams);
    unsigned capacity=info.max_entries,entries=1;
    while(entries<capacity+BGS_LLM_SLOTS+2)entries*=2;
    net->sq_bytes=io_align((size_t)entries*sizeof(io_uring_sqe));
    net->ring_bytes=io_align(4096+(size_t)entries*2*sizeof(io_uring_cqe));
    bgs_ktcp_layout &layout=net->layout;
    layout.capacity=capacity;layout.entries=entries;layout.ring_bytes=net->ring_bytes;
    size_t cursor=(sizeof(bgs_ktcp_shared)+63)&~size_t(63);
    layout.conns=cursor;cursor+=(size_t)capacity*sizeof(bgs_ktcp_conn);
    cursor=(cursor+63)&~size_t(63);
    layout.buffers=cursor;cursor+=(size_t)capacity*BGS_KTCP_RX_STRIDE;
    layout.tx=cursor;cursor+=(size_t)BGS_LLM_SLOTS*BGS_KTCP_TX_STRIDE;
    layout.tick=cursor;cursor+=sizeof(__kernel_timespec);
    net->mem_bytes=io_align(cursor);
    if(net->mem_bytes>UINT32_MAX){
        fprintf(stderr,"kernel TCP registered layout exceeds32-bit offsets\n");return 1;
    }
    layout.mem_bytes=net->mem_bytes;layout.listener=listener;
    net->sq=kernel_memory(net->sq_bytes);net->rings=kernel_memory(net->ring_bytes);
    net->memory=kernel_memory(net->mem_bytes);
    if(!net->sq||!net->rings||!net->memory)return 1;
    layout.user_base=(uint64_t)net->memory;
    net->report=(bgs_ktcp_shared*)net->memory;
    auto *s=net->report;s->free=0;s->active=-1;s->accepting=-1;s->timeout_active=1;
    auto *conns=(bgs_ktcp_conn*)((char*)net->memory+layout.conns);
    for(unsigned i=0;i<capacity;i++){
        conns[i].fd=-1;conns[i].next=i+1<capacity?(int)i+1:-1;
        conns[i].reply=-1;conns[i].last=-1;
    }
    for(auto &request:s->requests){request.conn=-1;request.next=-1;}
    int identity=bpf_obj_get("/sys/fs/bpf/bgs_uring_owner");unsigned key=0;
    struct stat namespace_info{};
    if(stat("/proc/self/ns/pid",&namespace_info)){perror("kernel TCP PID namespace");return 2;}
    bgs_uring_context owner{(unsigned)getpid(),0,0,
                         (unsigned long long)namespace_info.st_dev,
                         (unsigned long long)namespace_info.st_ino};
    if(identity<0||bpf_map_update_elem(identity,&key,&owner,BPF_ANY)){
        if(identity>=0)close(identity);perror("kernel TCP identity");return 2;
    }
    io_uring_params parameters{};
    parameters.flags=IORING_SETUP_NO_MMAP|IORING_SETUP_NO_SQARRAY|
                     IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN;
    parameters.sq_off.user_addr=(uint64_t)net->sq;
    parameters.cq_off.user_addr=(uint64_t)net->rings;
    net->ringfd=syscall(__NR_io_uring_setup,entries,&parameters);
    if(net->ringfd<0){close(identity);perror("kernel TCP setup");return 3;}
    if(parameters.cq_off.cqes+(size_t)parameters.cq_entries*sizeof(io_uring_cqe)>
       net->ring_bytes || parameters.sq_entries!=entries ||
       parameters.cq_entries!=2*entries){close(identity);return 4;}
    if(bpf_map_lookup_elem(identity,&key,&owner)||!owner.ring){close(identity);return 5;}
    owner.pad=1;
    if(bpf_map_update_elem(identity,&key,&owner,BPF_ANY)){close(identity);return 5;}
    close(identity);
    layout.sh=parameters.sq_off.head;layout.st=parameters.sq_off.tail;
    layout.ch=parameters.cq_off.head;layout.ct=parameters.cq_off.tail;
    layout.cqes=parameters.cq_off.cqes;
    layout.dropped=parameters.sq_off.dropped;
    layout.overflow=parameters.cq_off.overflow;
    s->loop.sq_head=layout.sh;s->loop.sq_tail=layout.st;s->loop.cq_tail=layout.ct;
    auto *sqes=(io_uring_sqe*)net->sq;
    auto *head=(unsigned*)((char*)net->rings+layout.ch);
    auto *tail=(unsigned*)((char*)net->rings+layout.ct);
    auto *sqtail=(unsigned*)((char*)net->rings+layout.st);
    sqes[0].opcode=IORING_OP_NOP;sqes[0].user_data=0xb007;
    __atomic_store_n(sqtail,1,__ATOMIC_RELEASE);
    if(syscall(__NR_io_uring_enter,net->ringfd,1,1,IORING_ENTER_GETEVENTS,nullptr,0)<0)return 6;
    while(!__atomic_load_n(tail,__ATOMIC_ACQUIRE))usleep(1000);
    auto *cqes=(io_uring_cqe*)((char*)net->rings+layout.cqes);
    if(cqes[0].user_data!=0xb007||cqes[0].res)return 7;
    __atomic_store_n(head,1,__ATOMIC_RELEASE);
    auto *tick=(__kernel_timespec*)((char*)net->memory+layout.tick);
    tick->tv_sec=0;tick->tv_nsec=60000;
    sqes[1]={};sqes[1].opcode=IORING_OP_TIMEOUT;
    sqes[1].addr=(uint64_t)tick;sqes[1].len=1;
    sqes[1].timeout_flags=IORING_TIMEOUT_MULTISHOT;sqes[1].user_data=8ULL<<56;
    s->pending=1;
    __atomic_store_n(sqtail,2,__ATOMIC_RELEASE);
    // First BPF step submits this timer with its initial ACCEPT. Do not start
    // periodic CQ production during potentially slow first verifier/JIT load.
    if(kernel_attach(net))return 9;
    printf("network=kernel BPF TCP controller; no GPU I/O CTA\n"
           "NET_READY pid=%d port=%u ring_bound=1 capacity=%u sq=%u cq=%u\n",
           getpid(),BGS_LLM_TCP_PORT,capacity,parameters.sq_entries,parameters.cq_entries);
    fflush(stdout);return 0;
}
static int uring_finish(UringNetwork *net){
    auto &s=*net->report;
    printf("ADMISSION queued_peak=%u\n",s.loop.backlog_peak);
    printf("RESULT {\"accepted\":%u,\"received\":%u,\"sends\":%u,\"bytes\":%u,"
           "\"completed\":%u,\"closed\":%u,\"aborted\":%u,"
           "\"max_connections\":%u,\"max_requests\":%u,\"pending\":%u,"
           "\"error\":%u,\"head\":%u,\"reserved\":%u,\"dropped\":%u,"
           "\"overflow\":%u,\"sqhead\":%u,\"sqtail\":%u,\"cqhead\":%u,"
           "\"cqtail\":%u,\"trace_count\":%u,\"trace_overflow\":%u}\n",
           s.accepted,s.received,s.sends,s.bytes,s.completed,s.closed,s.aborted,
           s.max_connections,s.max_requests,s.pending,s.error,s.head,s.reserved,
           s.dropped,s.overflow,s.sqhead,s.sqtail,s.cqhead,s.cqtail,
           s.trace_count,s.trace_overflow);
    for(unsigned i=0;i<s.trace_count&&i<128;i++){
        const auto &r=s.records[i];
        printf("REPLY {\"seq\":%u,\"gen\":%u,\"sends\":%u,\"early\":%u,"
               "\"bytes\":%u,\"aborted\":%u,\"client_ns\":%llu}\n",
               r.seq,r.gen,r.sends,r.early,r.bytes,r.aborted,r.client_ns);
    }
    bool failed=s.pending||s.error||s.dropped||s.overflow||s.active>=0||
                s.connections||!s.loop.finished||s.sqhead!=s.sqtail||s.cqhead!=s.cqtail;
    io_signal_control=nullptr;io_loop_tid=0;
    bpf_link__destroy(net->loop_link);bpf_object__close(net->loop_object);
    close(net->ringfd);
    munmap(net->sq,net->sq_bytes);munmap(net->rings,net->ring_bytes);
    munmap(net->memory,net->mem_bytes);
    printf("CLEANUP ring_connections_buffers_released\n");fflush(stdout);
    return failed;
}
#endif
