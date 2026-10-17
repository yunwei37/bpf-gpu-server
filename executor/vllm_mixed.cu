// SPDX-License-Identifier: Apache-2.0
// Kernel/shared-queue serving with upstream model graphs and GPU-owned metadata.
#include <cuda_runtime.h>
#include <bpf/bpf.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <vector>
#include "bpf_gpu_server_queue.h"
#define CUDA(x) do { cudaError_t e=(x); if(e) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return (int)e; } } while(0)
#ifdef BGS_USER_NETWORK
#include "userspace_tcp.h"
#endif
#ifdef BGS_URING_NETWORK
#ifdef BGS_KERNEL_TCP
#include "kernel_tcp.h"
#else
#include "io_uring_tcp.h"
#endif
#endif
struct Step { cudaGraphExec_t graph; long long* output; };
struct Inputs {
    int *ids,*lens,*starts,*table,*pages;
    long long *positions,*slots;
    bool* padding;
    unsigned stride,block_size,pages_per_request;
};
struct Request { unsigned id,prompt,gen,k,position,last,phase; };
struct Trace {
    unsigned kind,id,query,batch;
    union { unsigned gen; unsigned input_tokens; };
    unsigned bad;
    unsigned long long client_ns;
    unsigned requests[BGS_LLM_SLOTS],tokens[BGS_LLM_MAX_TOK];
};
struct Control {
    unsigned seen,retired,reclaim,error,launches,device_launch_calls,max_active,batch,query,release_rounds,max_release;
    unsigned input_tokens;
    unsigned selected[BGS_LLM_SLOTS];
    Request requests[BGS_LLM_SLOTS];
    Trace* trace;
    unsigned trace_capacity,trace_count,trace_overflows;
};
__device__ unsigned queue_load(const unsigned* p) {
    unsigned value;asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");return value;
}
__device__ unsigned long long queue_load(const unsigned long long* p) {
    unsigned long long value;asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");return value;
}
__device__ Trace* trace_next(Control* c) {
#ifdef BGS_TRACE
    unsigned index=c->trace_count++;
    if(index<c->trace_capacity)return &c->trace[index];
    c->trace_overflows++;
#endif
    return nullptr;
}
__global__ void dispatch(bgs_page* p,Step* steps,Inputs in,unsigned vocab,Control* c) {
    if(threadIdx.x || blockIdx.x)return;
    for(;;) {
        if(c->batch) {
            Trace* event=trace_next(c);
            if(event) { event->kind=2;event->query=c->query;event->batch=c->batch;event->input_tokens=c->input_tokens; }
            Step* step=&steps[c->input_tokens];
            for(unsigned row=0;row<c->batch;row++) {
                unsigned i=c->selected[row];Request* r=&c->requests[i];
                unsigned token=(unsigned)step->output[row];
                if(event) { event->requests[row]=r->id;event->tokens[row]=token; }
                bgs_llm_slot* slot=&p->llm[i];
                __stcg(&slot->tok_out[r->k],token);__threadfence_system();__stcg(&slot->produced,r->k+1);
                r->last=token;r->k++;
                r->position=r->k==1 ? r->prompt : r->position+1;
                r->phase=r->k==r->gen ? 2 : 1;
            }
            c->batch=0;
        }
        if(queue_load((unsigned long long*)&p->stop_ns))return;
        unsigned release_begin=c->retired,i=c->retired%BGS_LLM_SLOTS;
#ifdef BGS_PREFIX_RELEASE
        // Model/head work is complete. TX owns each reservation until its
        // final slot access; releasing the FIFO prefix needs no global wait.
        while(c->retired!=c->seen && c->requests[i].phase==2) {
            __threadfence_system();__stcg(&p->llm[i].state,(unsigned)BGS_DONE);
            c->requests[i].phase=0;c->retired++;i=c->retired%BGS_LLM_SLOTS;
        }
#else
        if(c->reclaim && queue_load(&p->llm[(c->retired-1)%BGS_LLM_SLOTS].state)!=BGS_DONE)c->reclaim=0;
        if(!c->reclaim && c->retired!=c->seen && c->requests[i].phase==2) {
            __threadfence_system();__stcg(&p->llm[i].state,(unsigned)BGS_DONE);
            c->requests[i].phase=0;c->retired++;c->reclaim=1;
        }
#endif
        unsigned released=c->retired-release_begin;
        if(released) {
            c->release_rounds++;if(released>c->max_release)c->max_release=released;
            Trace* event=trace_next(c);
            if(event) {
                event->kind=3;event->batch=released;
                for(unsigned row=0;row<released;row++)event->requests[row]=release_begin+row;
            }
        }
#ifdef BGS_IO_COALESCED
        // Advance I/O after result publication and before model admission.
        // This CTA returns while model children execute.
        io_progress(io_state);
        // Drain completed responses and published I/O before fresh model work.
        if(c->retired==c->seen &&
           (io_state->next_final!=c->retired ||
            io_load(io_state->ring.head)!=io_load(io_state->ring.tail)))continue;
#endif
        while(queue_load(&p->llm_head)!=c->seen) {
            i=c->seen%BGS_LLM_SLOTS;bgs_llm_slot* slot=&p->llm[i];
            if(c->requests[i].phase || queue_load(&slot->state)!=BGS_PENDING)break;
            unsigned np=__ldcg(&slot->n_prompt),ng=__ldcg(&slot->n_gen);
            bool bad=!np || np>BGS_LLM_MAX_TOK || !ng || ng>BGS_LLM_MAX_TOK;
            for(unsigned j=0;j<np && j<BGS_LLM_MAX_TOK;j++)if(__ldcg(&slot->tok_in[j])>=vocab)bad=true;
            unsigned short flags=__ldcg(&slot->pad);
            __threadfence_system();
            __stcg(&slot->pad,(unsigned short)((flags&~(BGS_LLM_VALIDATING|BGS_LLM_REJECTED))|(bad ? BGS_LLM_REJECTED : 0)));
            c->requests[i]={c->seen,np,ng,0,0,0,bad ? 2u : 3u};
            Trace* event=trace_next(c);
            if(event) {
                event->kind=1;event->id=c->seen;event->query=np;event->gen=ng;event->bad=bad;
                event->client_ns=queue_load((unsigned long long*)&slot->client_ns);
                for(unsigned j=0;j<np && j<BGS_LLM_MAX_TOK;j++)event->tokens[j]=__ldcg(&slot->tok_in[j]);
            }
            c->seen++;
        }
        unsigned query=0;
        for(unsigned id=c->retired;id!=c->seen;id++) {
            Request* r=&c->requests[id%BGS_LLM_SLOTS];
            if(r->phase==3) {
#ifdef BGS_PAD_PREFILL
                if(r->prompt>query)query=r->prompt;
#else
                query=r->prompt;break;
#endif
            }
        }
        for(unsigned id=c->retired;id!=c->seen;id++) {
            i=id%BGS_LLM_SLOTS;Request* r=&c->requests[i];
            if(r->phase==3 || r->phase==1)c->selected[c->batch++]=i;
        }
        if(!c->batch)continue;
        unsigned count=0;
        for(unsigned row=0;row<c->batch;row++) {
            i=c->selected[row];Request* r=&c->requests[i];
            bool prefill=r->phase==3;
            unsigned length=prefill ? r->prompt : 1;
            in.lens[row]=prefill ? r->prompt : r->position+1;
            in.starts[row]=count;
            for(unsigned column=0;column<in.pages_per_request;column++)
                in.table[row*in.stride+column]=in.pages[i*in.pages_per_request+column];
            for(unsigned j=0;j<length;j++) {
                unsigned index=count+j,position=prefill ? j : r->position;
                in.ids[index]=prefill ? __ldcg(&p->llm[i].tok_in[j]) : r->last;
                in.positions[index]=position;in.padding[index]=false;
                in.slots[index]=(long long)in.pages[i*in.pages_per_request+position/in.block_size]*in.block_size+position%in.block_size;
            }
            count+=length;r->phase=4;
        }
        unsigned graph_reqs=count<BGS_LLM_SLOTS ? count : BGS_LLM_SLOTS;
        for(unsigned row=c->batch;row<graph_reqs;row++) {
            in.lens[row]=0;in.starts[row]=count;
            for(unsigned column=0;column<in.pages_per_request;column++)in.table[row*in.stride+column]=0;
        }
        in.starts[graph_reqs]=count;c->query=query;c->input_tokens=count;
        if(c->batch>c->max_active)c->max_active=c->batch;
        __threadfence();
        cudaError_t e=cudaGraphLaunch(steps[count].graph,
#ifdef BGS_CHAINED_DISPATCH
            cudaStreamGraphTailLaunch);
#else
            cudaStreamGraphFireAndForget);
#endif
        if(e) { c->error=e;return; }
        c->launches++;c->device_launch_calls++;
#ifndef BGS_CHAINED_DISPATCH
        e=cudaGraphLaunch(cudaGetCurrentGraphExec(),cudaStreamGraphTailLaunch);
        if(e)c->error=e;else c->device_launch_calls++;
#endif
        return;
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
extern "C" int serve(const unsigned long long* bodies,const unsigned long long* heads,
 const unsigned long long* outputs,const unsigned long long* pointers,unsigned stride,
 unsigned block_size,unsigned pages_per_request,unsigned vocab,unsigned layers,unsigned h,
 unsigned seconds,unsigned trace_requests,const char* trace_path) {
    unsigned size=BGS_LLM_MAX_TOK*BGS_LLM_SLOTS+1;
    std::vector<Step> host(size);std::vector<cudaGraph_t> combined(size);
    Step* steps;CUDA(cudaMalloc(&steps,size*sizeof(Step)));
    Inputs in={};
    in.ids=(int*)pointers[0];in.positions=(long long*)pointers[1];in.lens=(int*)pointers[2];
    in.starts=(int*)pointers[3];in.padding=(bool*)pointers[4];in.slots=(long long*)pointers[5];
    in.table=(int*)pointers[6];in.pages=(int*)pointers[7];in.stride=stride;in.block_size=block_size;in.pages_per_request=pages_per_request;
    Control initial={};
#ifdef BGS_TRACE
    Trace* trace=nullptr;
    if(!trace_requests || !trace_path || !*trace_path) { fprintf(stderr,"traced candidate requires bounded workload and output path\n");return 1; }
    initial.trace_capacity=trace_requests*(BGS_LLM_MAX_TOK+2);
    CUDA(cudaMalloc(&trace,initial.trace_capacity*sizeof(Trace)));initial.trace=trace;
#endif
    Control* control;CUDA(cudaMalloc(&control,sizeof(Control)));CUDA(cudaMemcpy(control,&initial,sizeof(initial),cudaMemcpyHostToDevice));
#ifdef BGS_USER_NETWORK
    bgs_page* page=(bgs_page*)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
#else
    int fd=bpf_obj_get("/sys/fs/bpf/bgs_ctl");if(fd<0) { perror("bpf_obj_get");return 1; }
    bgs_page* page=(bgs_page*)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
#endif
    if(page==MAP_FAILED) { perror("mmap");return 1; }
    CUDA(cudaHostRegister(page,BGS_PAGE_MMAP_BYTES,cudaHostRegisterMapped));
    bgs_page* device;CUDA(cudaHostGetDevicePointer((void**)&device,page,0));
    int listener=socket(AF_INET,SOCK_STREAM,0),yes=1;setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    if(setsockopt(listener,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes))) { perror("TCP_NODELAY");close(listener);return 1; }
    sockaddr_in addr={};addr.sin_family=AF_INET;addr.sin_port=htons(BGS_LLM_TCP_PORT);
    const char* bind_addr=getenv("BGS_BIND");
    if(inet_pton(AF_INET,bind_addr ? bind_addr : "127.0.0.1",&addr.sin_addr)!=1) { fprintf(stderr,"BGS_BIND must be an IPv4 address\n");return 1; }
    if(bind(listener,(sockaddr*)&addr,sizeof(addr)) || listen(listener,SOMAXCONN)) { perror("listen");return 1; }
    __atomic_store_n(&page->stop_ns,0,__ATOMIC_RELEASE);
    void* args[]={&device,&steps,&in,&vocab,&control};
    cudaKernelNodeParams params={};params.func=(void*)dispatch;params.gridDim=dim3(1);params.blockDim=dim3(32);params.kernelParams=args;
    for(unsigned index=1;index<size;index++) {
        cudaGraphNode_t body,head;
        CUDA(cudaGraphCreate(&combined[index],0));
        CUDA(cudaGraphAddChildGraphNode(&body,combined[index],nullptr,0,(cudaGraph_t)bodies[index]));
        CUDA(cudaGraphAddChildGraphNode(&head,combined[index],&body,1,(cudaGraph_t)heads[index]));
        CUDA(cudaGraphInstantiateWithFlags(&host[index].graph,combined[index],cudaGraphInstantiateFlagDeviceLaunch));
        CUDA(cudaGraphUpload(host[index].graph,0));host[index].output=(long long*)outputs[index];
    }
    CUDA(cudaMemcpy(steps,host.data(),size*sizeof(Step),cudaMemcpyHostToDevice));
    cudaGraph_t parent_graph;cudaGraphNode_t node;cudaGraphExec_t parent;
    CUDA(cudaGraphCreate(&parent_graph,0));CUDA(cudaGraphAddKernelNode(&node,parent_graph,nullptr,0,&params));
    CUDA(cudaGraphInstantiateWithFlags(&parent,parent_graph,cudaGraphInstantiateFlagDeviceLaunch));CUDA(cudaGraphUpload(parent,0));CUDA(cudaDeviceSynchronize());
#ifdef BGS_URING_NETWORK
    UringNetwork network{};
    if(uring_start(page,device,listener,&network))return 1;
    close(fd);
#elif !defined(BGS_USER_NETWORK)
    int module=open("module/bpf_gpu_server_tx.ko",O_RDONLY|O_CLOEXEC);if(module<0) { perror("module open");return 1; }
    char module_params[80];snprintf(module_params,sizeof(module_params),"map_fd=%d listen_fd=%d",fd,listener);
    if(syscall(SYS_finit_module,module,module_params,0)) { perror("finit_module");return 1; }
    close(module);close(fd);
#endif
    signal(SIGTERM,stop_signal);signal(SIGINT,stop_signal);
    cudaError_t launched=cudaGraphLaunch(parent,0);
    if(launched==cudaSuccess) {
        printf("resident vLLM ready: %u layers h=%u vocab=%u; tensor batches1..%u, GPU prefill/decode\n",layers,h,vocab,BGS_LLM_SLOTS);fflush(stdout);
#ifdef BGS_USER_NETWORK
        launched=user_network(page,listener,seconds,&stopped) ? cudaErrorUnknown : cudaSuccess;
#elif defined(BGS_URING_BPF_LOOP)
        launched=uring_wait(&network,seconds,&stopped) ? cudaErrorUnknown : cudaSuccess;
#else
        while(seconds-- && !stopped)sleep(1);
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
#ifdef BGS_TRACE
    unsigned count=result.trace_count<result.trace_capacity ? result.trace_count : result.trace_capacity;
    std::vector<Trace> records(count);CUDA(cudaMemcpy(records.data(),trace,count*sizeof(Trace),cudaMemcpyDeviceToHost));
    FILE* file=fopen(trace_path,"w");if(!file) { perror("trace output");return 1; }
    for(const auto& r:records) {
        if(r.kind==1) {
            fprintf(file,"{\"event\":\"admission\",\"id\":%u,\"query\":%u,\"gen\":%u,\"bad\":%u,\"client_ns\":%llu,\"input_ids\":[",r.id,r.query,r.gen,r.bad,r.client_ns);
            for(unsigned j=0;j<r.query && j<BGS_LLM_MAX_TOK;j++)fprintf(file,"%s%u",j ? "," : "",r.tokens[j]);
        }else if(r.kind==3) {
            fprintf(file,"{\"event\":\"release\",\"batch\":%u,\"request_ids\":[",r.batch);
            for(unsigned j=0;j<r.batch;j++)fprintf(file,"%s%u",j ? "," : "",r.requests[j]);
        }else {
            fprintf(file,"{\"event\":\"compute\",\"query\":%u,\"batch\":%u,\"input_tokens\":%u,\"request_ids\":[",r.query,r.batch,r.input_tokens);
            for(unsigned j=0;j<r.batch;j++)fprintf(file,"%s%u",j ? "," : "",r.requests[j]);
            fprintf(file,"],\"tokens\":[");
            for(unsigned j=0;j<r.batch;j++)fprintf(file,"%s%u",j ? "," : "",r.tokens[j]);
        }
        fprintf(file,"]}\n");
    }
    fprintf(file,"{\"event\":\"terminal\",\"records\":%u,\"overflows\":%u}\n",result.trace_count,result.trace_overflows);
    if(fclose(file)) { perror("trace close");return 1; }
    CUDA(cudaFree(trace));
#endif
    CUDA(cudaHostUnregister(page));munmap(page,BGS_PAGE_MMAP_BYTES);
    CUDA(cudaGraphExecDestroy(parent));CUDA(cudaGraphDestroy(parent_graph));CUDA(cudaFree(steps));CUDA(cudaFree(control));
    for(unsigned index=1;index<size;index++) { CUDA(cudaGraphExecDestroy(host[index].graph));CUDA(cudaGraphDestroy(combined[index])); }
    printf("resident vLLM stopped: launches=1 device_graph_launches=%u graph_error=%u max_active=%u admitted=%u retired=%u trace_overflows=%u release_rounds=%u max_release=%u device_launch_calls=%u\n",result.launches,result.error,result.max_active,result.seen,result.retired,result.trace_overflows,result.release_rounds,result.max_release,result.device_launch_calls);fflush(stdout);
    return result.error || result.trace_overflows;
}
