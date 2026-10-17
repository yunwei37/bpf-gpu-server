// SPDX-License-Identifier: Apache-2.0
// GPU-tail graph dispatch: host work is model/graph bootstrap and shutdown only.
#include <cuda_runtime.h>
#include <bpf/bpf.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstdlib>
#include <unistd.h>
#include <cstdio>
#include <csignal>
#include <vector>
#include "bpf_gpu_server_queue.h"
#define CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return (int)e; } } while(0)
#ifdef BGS_USER_NETWORK
#include "userspace_tcp.h"
#endif
#ifdef BGS_URING_NETWORK
#include "io_uring_tcp.h"
#endif
struct Step { cudaGraphExec_t graph; long long *output; };
struct Request { unsigned k,prompt,gen,step,phase; };
struct Control { unsigned seen,retired,reclaim,error,launches,max_active; Request requests[BGS_LLM_SLOTS]; };

// System-scope acquire protects host publication and forbids polling-load
// elimination; a cache modifier alone does not give synchronization semantics.
__device__ unsigned queue_load(const unsigned *p) {
    unsigned value;asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");return value;
}
__device__ unsigned long long queue_load(const unsigned long long *p) {
    unsigned long long value;asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");return value;
}

__global__ void dispatch(bgs_page *p,Step *steps,long long **inputs,unsigned count,unsigned vocab,Control *c,bool once) {
    if(threadIdx.x || blockIdx.x) return;
    while(!queue_load((unsigned long long *)&p->stop_ns)) {
        // A parent tail resumes only after all private child graphs finish.
        // phase1 awaits sampling; phase3 awaits launch, including after a
        // host-control return that publishes another request's DONE.
        for(unsigned i=0;i<BGS_LLM_SLOTS;i++) {
            Request *r=&c->requests[i];
            if(r->phase!=1) continue;
            unsigned bank=i;
            unsigned token=(unsigned)*steps[bank*count+r->step].output;
            bgs_llm_slot *slot=&p->llm[i];
            __stcg(&slot->tok_out[r->k],token);__threadfence_system();__stcg(&slot->produced,r->k+1);
            if(++r->k==r->gen) r->phase=2;
            else { inputs[bank][0]=token;r->step=BGS_LLM_MAX_TOK+r->prompt+r->k-1;r->phase=3; }
        }
        // The head-indexed ring must reclaim in admission order. Hold final
        // tokens until the previous DONE was reclaimed by kernel TX. A reused
        // WRITING/PENDING slot also proves that its old ownership was released.
        if(c->reclaim && queue_load(&p->llm[(c->retired-1)%BGS_LLM_SLOTS].state)!=BGS_DONE) c->reclaim=0;
        unsigned idx=c->retired%BGS_LLM_SLOTS;
        if(!c->reclaim && c->retired!=c->seen && c->requests[idx].phase==2) {
            __threadfence_system();__stcg(&p->llm[idx].state,(unsigned)BGS_DONE);
            c->requests[idx].phase=0;c->retired++;c->reclaim=1;
            if(once) return;
        }
        // Admission remains sequential; computation uses each physical slot's
        // private KV/input/output bank. No producer or page ABI change.
#ifdef BGS_IO_COALESCED
        // Advance I/O after result publication and before model admission.
        // This CTA returns while model children execute.
        io_progress(io_state);
#endif
        while(queue_load(&p->llm_head)!=c->seen) {
            idx=c->seen%BGS_LLM_SLOTS;
            bgs_llm_slot *slot=&p->llm[idx];
            if(c->requests[idx].phase || queue_load(&slot->state)!=BGS_PENDING) break;
            unsigned bank=idx;
            unsigned np=__ldcg(&slot->n_prompt),ng=__ldcg(&slot->n_gen);
            bool bad=!np || np>BGS_LLM_MAX_TOK || !ng || ng>BGS_LLM_MAX_TOK;
            for(unsigned j=0;j<np && j<BGS_LLM_MAX_TOK;j++) {
                unsigned token=__ldcg(&slot->tok_in[j]);
                if(token>=vocab) bad=true;
                inputs[bank][j]=token;
            }
            unsigned short flags=__ldcg(&slot->pad);
            __threadfence_system();
            __stcg(&slot->pad,(unsigned short)((flags&~BGS_LLM_VALIDATING)|(bad ? BGS_LLM_REJECTED : 0)));
            c->requests[idx]={0,np,ng,np,bad ? 2u : 3u};c->seen++;
        }
        unsigned active=0;
        for(unsigned i=0;i<BGS_LLM_SLOTS;i++) {
            Request *r=&c->requests[i];
            if(r->phase!=3) continue;
            unsigned bank=i;
            cudaError_t e=cudaGraphLaunch(steps[bank*count+r->step].graph,cudaStreamGraphFireAndForget);
            if(e!=cudaSuccess) { c->error=e;return; }
            r->phase=1;c->launches++;active++;
        }
        if(active) {
            if(active>c->max_active) c->max_active=active;
            cudaError_t e=cudaGraphLaunch(cudaGetCurrentGraphExec(),cudaStreamGraphTailLaunch);
            if(e!=cudaSuccess) c->error=e;
            return;
        }
        // Idle, rejection and TX reclamation waits stay on the GPU.
    }
}
extern "C" unsigned max_tokens() { return BGS_LLM_MAX_TOK; }
extern "C" unsigned max_requests() { return BGS_LLM_SLOTS; }
static volatile sig_atomic_t stopped=0;
static void stop_signal(int) {
    stopped=1;
#ifdef BGS_USER_NETWORK
    user_network_interrupt();
#endif
#ifdef BGS_URING_BPF_LOOP
    uring_interrupt();
#endif
}
extern "C" int serve(const unsigned long long *graphs,const unsigned long long *outputs,unsigned count,const unsigned long long *input_ptrs,unsigned vocab,unsigned layers,unsigned h,unsigned seconds) {
    unsigned banks=max_requests(),size=count*banks;
    std::vector<Step> host(size);
    for(unsigned i=0;i<size;i++) {
        if(i%count==0) continue;
        CUDA(cudaGraphInstantiate(&host[i].graph,(cudaGraph_t)graphs[i],cudaGraphInstantiateFlagDeviceLaunch));
        CUDA(cudaGraphUpload(host[i].graph,0));host[i].output=(long long *)outputs[i];
    }
    Step *steps;CUDA(cudaMalloc(&steps,size*sizeof(Step)));CUDA(cudaMemcpy(steps,host.data(),size*sizeof(Step),cudaMemcpyHostToDevice));
    Control *control;CUDA(cudaMalloc(&control,sizeof(Control)));CUDA(cudaMemset(control,0,sizeof(Control)));
#ifdef BGS_USER_NETWORK
    bgs_page *page=(bgs_page *)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
#else
    int fd=bpf_obj_get("/sys/fs/bpf/bgs_ctl");if(fd<0) { perror("bpf_obj_get");return 1; }
    bgs_page *page=(bgs_page *)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
#endif
    if(page==MAP_FAILED) { perror("mmap");return 1; }
    CUDA(cudaHostRegister(page,BGS_PAGE_MMAP_BYTES,cudaHostRegisterMapped));
    bgs_page *device;CUDA(cudaHostGetDevicePointer((void **)&device,page,0));
    int listener=socket(AF_INET,SOCK_STREAM,0),yes=1;setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in addr={};addr.sin_family=AF_INET;addr.sin_port=htons(BGS_LLM_TCP_PORT);
    const char *bind_addr=getenv("BGS_BIND");
    if(inet_pton(AF_INET,bind_addr ? bind_addr : "127.0.0.1",&addr.sin_addr)!=1) {
        fprintf(stderr,"BGS_BIND must be an IPv4 address\n");return 1;
    }
    if(bind(listener,(sockaddr *)&addr,sizeof(addr)) || listen(listener,SOMAXCONN)) { perror("listen");return 1; }
    __atomic_store_n(&page->stop_ns,0,__ATOMIC_RELEASE);
    long long **inputs;CUDA(cudaMalloc(&inputs,banks*sizeof(long long *)));
    CUDA(cudaMemcpy(inputs,input_ptrs,banks*sizeof(long long *),cudaMemcpyHostToDevice));
#ifdef BGS_HOST_LAUNCH
    bool once=true;const char *mode="host-launch";
#else
    bool once=false;const char *mode="resident";
#endif
    void *args[]={&device,&steps,&inputs,&count,&vocab,&control,&once};
    cudaKernelNodeParams params={};params.func=(void *)dispatch;params.gridDim=dim3(1);params.blockDim=dim3(32);params.kernelParams=args;
    cudaGraph_t graph;CUDA(cudaGraphCreate(&graph,0));cudaGraphNode_t node;CUDA(cudaGraphAddKernelNode(&node,graph,nullptr,0,&params));
    cudaGraphExec_t parent;CUDA(cudaGraphInstantiate(&parent,graph,cudaGraphInstantiateFlagDeviceLaunch));CUDA(cudaGraphUpload(parent,0));CUDA(cudaDeviceSynchronize());
#ifdef BGS_URING_NETWORK
    UringNetwork network{};
    if(uring_start(page,device,listener,&network))return 1;
    close(fd);
#elif !defined(BGS_USER_NETWORK)
    int module=open("module/bpf_gpu_server_tx.ko",O_RDONLY|O_CLOEXEC);if(module<0) { perror("module open");return 1; }
    char params_module[80];snprintf(params_module,sizeof(params_module),"map_fd=%d listen_fd=%d",fd,listener);
    if(syscall(SYS_finit_module,module,params_module,0)) { perror("finit_module");return 1; }
    close(module);close(fd);
#endif
    signal(SIGTERM,stop_signal);signal(SIGINT,stop_signal);
#ifdef BGS_USER_NETWORK
    printf("network=userspace epoll; same resident GPU graph dispatch\n");fflush(stdout);
#endif
    cudaError_t launched=cudaSuccess;unsigned launches=0;
    if(!once) { launched=cudaGraphLaunch(parent,0);launches++; }
    if(launched==cudaSuccess) {
        printf("resident Qwen ready: %u layers h=%u vocab=%u; device-tail graphs, dispatch=%s\n",layers,h,vocab,mode);fflush(stdout);
#ifdef BGS_USER_NETWORK
        launched=user_network(page,listener,seconds,&stopped) ? cudaErrorUnknown : cudaSuccess;
#elif defined(BGS_URING_BPF_LOOP)
        launched=uring_wait(&network,seconds,&stopped) ? cudaErrorUnknown : cudaSuccess;
#else
        if(once) {
            signal(SIGALRM,stop_signal);alarm(seconds);unsigned retired=0;
            while(!stopped) {
                if(__atomic_load_n(&page->llm_head,__ATOMIC_ACQUIRE)==retired ||
                   __atomic_load_n(&page->llm[retired%BGS_LLM_SLOTS].state,__ATOMIC_ACQUIRE)!=BGS_PENDING) continue;
                launched=cudaGraphLaunch(parent,0);launches++;
                if(launched!=cudaSuccess) break;
                launched=cudaDeviceSynchronize();if(launched!=cudaSuccess) break;
                retired++;
            }
            alarm(0);
        } else while(seconds-- && !stopped) sleep(1);
#endif
    }
    __atomic_store_n(&page->stop_ns,1,__ATOMIC_RELEASE);
#ifdef BGS_URING_NETWORK
    cudaError_t finished=cudaStreamSynchronize(0);CUDA(finished);
    uring_quiesce(&network);
#ifdef BGS_URING_BPF_LOOP
    if(uring_drain(&network))return 1;
#endif
    finished=cudaDeviceSynchronize();
    if(uring_finish(&network))return 1;
#else
    cudaError_t finished=cudaDeviceSynchronize();
#ifndef BGS_USER_NETWORK
    if(syscall(SYS_delete_module,"bpf_gpu_server_tx",0)) { perror("delete_module");return 1; }
#endif
#endif
    close(listener);CUDA(launched);CUDA(finished);
    Control result;CUDA(cudaMemcpy(&result,control,sizeof(result),cudaMemcpyDeviceToHost));
    CUDA(cudaHostUnregister(page));munmap(page,BGS_PAGE_MMAP_BYTES);
    CUDA(cudaGraphExecDestroy(parent));CUDA(cudaGraphDestroy(graph));CUDA(cudaFree(steps));CUDA(cudaFree(inputs));CUDA(cudaFree(control));
    for(unsigned i=0;i<size;i++) if(i%count) CUDA(cudaGraphExecDestroy(host[i].graph));
    printf("resident Qwen stopped: dispatch=%s launches=%u device_graph_launches=%u graph_error=%u banks=%u max_active=%u\n",mode,launches,result.launches,result.error,banks,result.max_active);fflush(stdout);
    return result.error;
}
