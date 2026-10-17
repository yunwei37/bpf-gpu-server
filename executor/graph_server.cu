// SPDX-License-Identifier: Apache-2.0
// A model-independent adapter for caller-owned, captured CUDA graphs.
#include <cuda_runtime.h>
#include <bpf/bpf.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <vector>
#include "bpf_gpu_server_queue.h"
#include "graph_backend.h"
#ifdef BGS_USER_NETWORK
#include "userspace_tcp.h"
#elif defined(BGS_KERNEL_TCP)
#include "kernel_tcp.h"
#else
#include "io_uring_tcp.h"
#endif
#define CUDA(x) do { cudaError_t e=(x); if(e) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return (int)e; } } while(0)
struct GraphSlot { cudaGraphExec_t graph; unsigned *input,*output; };
struct GraphControl {
    unsigned seen,retired,phase[BGS_LLM_SLOTS],inflight[BGS_LLM_SLOTS],count,error,launches;
};
__device__ unsigned graph_load(const unsigned* p) {
    unsigned v;asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ unsigned long long graph_load(const unsigned long long* p) {
    unsigned long long v;asm volatile("ld.acquire.sys.global.u64 %0, [%1];":"=l"(v):"l"(p):"memory");return v;
}
__global__ void graph_dispatch(bgs_page* p,GraphSlot* graphs,GraphControl* c,unsigned ni,unsigned no) {
    if(threadIdx.x || blockIdx.x)return;
    for(;;) {
        // A tail launch begins only after every fire-and-forget graph in the
        // preceding invocation has completed, including all its output writes.
        for(unsigned j=0;j<c->count;j++) {
            unsigned i=c->inflight[j];bgs_llm_slot* s=&p->llm[i];
            for(unsigned k=0;k<no;k++)__stcg(&s->tok_out[k],graphs[i].output[k]);
            __threadfence_system();__stcg(&s->produced,no);c->phase[i]=2;
        }
        c->count=0;
        while(c->retired!=c->seen && c->phase[c->retired%BGS_LLM_SLOTS]==2) {
            unsigned i=c->retired%BGS_LLM_SLOTS;
            __threadfence_system();__stcg(&p->llm[i].state,(unsigned)BGS_DONE);
            c->phase[i]=0;c->retired++;
        }
        if(graph_load((unsigned long long*)&p->stop_ns))return;
        while(graph_load(&p->llm_head)!=c->seen) {
            unsigned i=c->seen%BGS_LLM_SLOTS;bgs_llm_slot* s=&p->llm[i];
            if(c->phase[i] || graph_load(&s->state)!=BGS_PENDING)break;
            bool bad=__ldcg(&s->n_prompt)!=ni || __ldcg(&s->n_gen)!=no;
            unsigned short flags=__ldcg(&s->pad);
            for(unsigned k=0;k<ni && !bad;k++)graphs[i].input[k]=__ldcg(&s->tok_in[k]);
            __threadfence_system();
            __stcg(&s->pad,(unsigned short)((flags&~(BGS_LLM_VALIDATING|BGS_LLM_REJECTED))|
                                          (bad ? BGS_LLM_REJECTED : 0)));
            c->phase[i]=bad ? 2 : 1;c->seen++;
            if(!bad) {
                __threadfence();
                cudaError_t e=cudaGraphLaunch(graphs[i].graph,cudaStreamGraphFireAndForget);
                if(e) { c->error=e;return; }
                c->inflight[c->count++]=i;c->launches++;
            }
        }
        if(c->count) {
            cudaError_t e=cudaGraphLaunch(cudaGetCurrentGraphExec(),cudaStreamGraphTailLaunch);
            if(e)c->error=e;
            return;
        }
    }
}
extern "C" unsigned bgs_graph_abi_version() { return 1; }
extern "C" unsigned bgs_graph_max_slots() { return BGS_LLM_SLOTS; }
extern "C" unsigned bgs_graph_max_words() { return BGS_LLM_MAX_TOK; }
static volatile sig_atomic_t graph_stopped=0;
static void graph_stop(int) {
    graph_stopped=1;
#ifdef BGS_USER_NETWORK
    user_network_interrupt();
#else
    uring_interrupt();
#endif
}
extern "C" int bgs_graph_serve(const uint64_t* bodies,const uint64_t* inputs,
 const uint64_t* outputs,unsigned ni,unsigned no,unsigned seconds) {
    if(!bodies || !inputs || !outputs || !ni || ni>BGS_LLM_MAX_TOK ||
       !no || no>BGS_LLM_MAX_TOK) { fprintf(stderr,"invalid graph buffers/counts\n");return 1; }
    graph_stopped=0;
    std::vector<GraphSlot> host(BGS_LLM_SLOTS);
    for(unsigned i=0;i<BGS_LLM_SLOTS;i++) {
        if(!bodies[i] || !inputs[i] || !outputs[i])return 1;
        CUDA(cudaGraphInstantiateWithFlags(&host[i].graph,(cudaGraph_t)bodies[i],cudaGraphInstantiateFlagDeviceLaunch));
        CUDA(cudaGraphUpload(host[i].graph,0));
        host[i].input=(unsigned*)inputs[i];host[i].output=(unsigned*)outputs[i];
    }
    GraphSlot* graphs;GraphControl* control;
    CUDA(cudaMalloc(&graphs,host.size()*sizeof(GraphSlot)));
    CUDA(cudaMemcpy(graphs,host.data(),host.size()*sizeof(GraphSlot),cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&control,sizeof(GraphControl)));CUDA(cudaMemset(control,0,sizeof(GraphControl)));
#ifdef BGS_USER_NETWORK
    bgs_page* page=(bgs_page*)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
#else
    int fd=bpf_obj_get("/sys/fs/bpf/bgs_ctl");if(fd<0) { perror("bpf_obj_get");return 1; }
    bgs_page* page=(bgs_page*)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
#endif
    if(page==MAP_FAILED) { perror("mmap");return 1; }
    CUDA(cudaHostRegister(page,BGS_PAGE_MMAP_BYTES,cudaHostRegisterMapped));
    bgs_page* device;CUDA(cudaHostGetDevicePointer((void**)&device,page,0));
    int listener=socket(AF_INET,SOCK_STREAM,0),yes=1;
    setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    if(setsockopt(listener,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes))) { perror("TCP_NODELAY");close(listener);return 1; }
    sockaddr_in addr={};addr.sin_family=AF_INET;addr.sin_port=htons(BGS_LLM_TCP_PORT);
    const char* bind_addr=getenv("BGS_BIND");
    if(inet_pton(AF_INET,bind_addr ? bind_addr : "127.0.0.1",&addr.sin_addr)!=1)return 1;
    if(bind(listener,(sockaddr*)&addr,sizeof(addr)) || listen(listener,SOMAXCONN)) { perror("listen");return 1; }
    __atomic_store_n(&page->stop_ns,0,__ATOMIC_RELEASE);
    void* args[]={&device,&graphs,&control,&ni,&no};
    cudaKernelNodeParams params={};params.func=(void*)graph_dispatch;
    params.gridDim=dim3(1);params.blockDim=dim3(32);params.kernelParams=args;
    cudaGraph_t parent_graph;cudaGraphNode_t node;cudaGraphExec_t parent;
    CUDA(cudaGraphCreate(&parent_graph,0));
    CUDA(cudaGraphAddKernelNode(&node,parent_graph,nullptr,0,&params));
    CUDA(cudaGraphInstantiateWithFlags(&parent,parent_graph,cudaGraphInstantiateFlagDeviceLaunch));
    CUDA(cudaGraphUpload(parent,0));CUDA(cudaDeviceSynchronize());
#ifndef BGS_USER_NETWORK
    UringNetwork network{};
    if(uring_start(page,device,listener,&network))return 1;
    close(fd);
#endif
    auto old_term=signal(SIGTERM,graph_stop),old_int=signal(SIGINT,graph_stop);
    cudaError_t launched=cudaGraphLaunch(parent,0);
    if(launched==cudaSuccess) {
        printf("graph backend ready: abi=1 slots=%u input_words=%u output_words=%u\n",BGS_LLM_SLOTS,ni,no);fflush(stdout);
#ifdef BGS_USER_NETWORK
        if(user_network(page,listener,seconds,&graph_stopped))launched=cudaErrorUnknown;
#else
        if(uring_wait(&network,seconds,&graph_stopped))launched=cudaErrorUnknown;
#endif
    }
    __atomic_store_n(&page->stop_ns,1,__ATOMIC_RELEASE);
#ifndef BGS_USER_NETWORK
    CUDA(cudaStreamSynchronize(0));uring_quiesce(&network);
    if(uring_drain(&network))return 1;
    CUDA(cudaDeviceSynchronize());if(uring_finish(&network))return 1;
#else
    CUDA(cudaDeviceSynchronize());
#endif
    signal(SIGTERM,old_term);signal(SIGINT,old_int);close(listener);
    GraphControl result;CUDA(cudaMemcpy(&result,control,sizeof(result),cudaMemcpyDeviceToHost));
    CUDA(cudaGraphExecDestroy(parent));CUDA(cudaGraphDestroy(parent_graph));
    for(auto& s:host)CUDA(cudaGraphExecDestroy(s.graph));
    CUDA(cudaFree(graphs));CUDA(cudaFree(control));
    CUDA(cudaHostUnregister(page));munmap(page,BGS_PAGE_MMAP_BYTES);
    printf("graph backend stopped: admitted=%u retired=%u graph_launches=%u graph_error=%u\n",
           result.seen,result.retired,result.launches,result.error);fflush(stdout);
    CUDA(launched);return result.error;
}
