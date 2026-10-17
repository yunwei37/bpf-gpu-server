// SPDX-License-Identifier: Apache-2.0
// Correctness-first resident Qwen2 reference: a cooperative grid executes prefill,
// decode, KV attention and greedy sampling, directly on the pinned BPF page.
// Matrix prefill and Tensor Core linear layers; no per-request host launch/copy/sampling.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <bpf/bpf.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <csignal>
#include "bpf_gpu_server_queue.h"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
constexpr int CTX = 2 * BGS_LLM_MAX_TOK;
struct Config { unsigned h, inter, layers, heads, kvheads, vocab; float eps, theta; };
struct Layer { half *norm, *q, *qb, *k, *kb, *v, *vb, *o, *post, *gate, *up, *down; };
struct Model { Config c; half *embed, *norm; Layer *layer; float *inverse; };
struct Scratch { float *x, *norm, *q, *k, *v, *att, *out, *gate, *up, *prob, *logits, *keys, *values; unsigned *control, *input; };

__device__ unsigned rank() { return blockIdx.x*blockDim.x+threadIdx.x; }
__device__ unsigned threads() { return gridDim.x*blockDim.x; }
__device__ void sync_grid() { cooperative_groups::this_grid().sync(); }
__device__ float fp16(float x) { return __half2float(__float2half_rn(x)); }
__device__ float warp_sum(float v) {
    for (int d=16; d; d/=2) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}
__device__ void matvec(float *out, const half *w, const half *bias,
                      const float *x, int rows, int cols, int count=1) {
    int lane=threadIdx.x%32, warp=rank()/32;
    // A tile contains sixteen model rows and sixteen prompt-token columns.
    // Eight warps partition K; count=1 preserves replicated-vector decode.
    if(rows%16==0 && cols%16==0 && reinterpret_cast<uintptr_t>(w)%32==0) {
        __shared__ __align__(32) half input[8][256];
        __shared__ __align__(32) float output[8][256];
        int local=threadIdx.x/32, row_tiles=rows/16;
        for(int tile=blockIdx.x;tile<row_tiles*((count+15)/16);tile+=gridDim.x) {
            int row=tile%row_tiles*16, token=tile/row_tiles*16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::col_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc;
            nvcuda::wmma::fill_fragment(acc,0.0f);
            for(int col=local*16;col<cols;col+=8*16) {
                for(int i=lane;i<256;i+=32) {
                    int t=count==1 ? 0 : token+i/16;
                    input[local][i]=__float2half_rn(t<count ? x[t*cols+col+i%16] : 0.0f);
                }
                __syncwarp();
                nvcuda::wmma::load_matrix_sync(a,w+(size_t)row*cols+col,cols);
                nvcuda::wmma::load_matrix_sync(b,input[local],16);
                nvcuda::wmma::mma_sync(acc,a,b,acc);
                __syncwarp();
            }
            nvcuda::wmma::store_matrix_sync(output[local],acc,16,nvcuda::wmma::mem_row_major);
            __syncthreads();
            int r=threadIdx.x/16,t=token+threadIdx.x%16;
            if(t<count) {
                float sum=0;
                for(int part=0;part<8;part++) sum+=output[part][threadIdx.x];
                out[t*rows+row+r]=fp16(sum+(bias ? __half2float(bias[row+r]) : 0));
            }
            __syncthreads();
        }
    } else {
        for(int item=warp;item<count*rows;item+=threads()/32) {
            int row=item%rows,t=item/rows;
            float sum=0;
            for(int col=lane;col<cols;col+=32)
                sum=fmaf(__half2float(w[(size_t)row*cols+col]),x[t*cols+col],sum);
            sum=warp_sum(sum);
            if(!lane) out[item]=fp16(sum+(bias ? __half2float(bias[row]) : 0));
        }
    }
    sync_grid();
}
__device__ void rms(float *out,const float *x,const half *w,int n,float eps,int count=1) {
    // One CTA reduces each token; stride also covers devices with fewer SMs.
    for(unsigned t=blockIdx.x;t<(unsigned)count;t+=gridDim.x) {
        __shared__ float sums[8],inv;
        int base=t*n;
        float sum=0;
        for(int i=threadIdx.x;i<n;i+=blockDim.x) sum+=x[base+i]*x[base+i];
        sum=warp_sum(sum);
        if(!(threadIdx.x%32)) sums[threadIdx.x/32]=sum;
        __syncthreads();
        if(!threadIdx.x) { sum=0;for(int i=0;i<8;i++) sum+=sums[i];inv=rsqrtf(sum/n+eps); }
        __syncthreads();
        for(int i=threadIdx.x;i<n;i+=blockDim.x) out[base+i]=fp16(fp16(x[base+i]*inv)*__half2float(w[i]));
        __syncthreads();
    }
    sync_grid();
}
__device__ void rope(float *a,float *tmp,int n,int dim,int pos,const float *inverse,int count) {
    for(int i=rank();i<count*n;i+=threads()) {
        int d=i%dim,base=i-d,j=d%(dim/2);
        float angle=(pos+i/n)*inverse[j];
        float cs=fp16(cosf(angle)),sn=fp16(sinf(angle));
        float rot=d<dim/2 ? -a[base+d+dim/2] : a[base+d-dim/2];
        tmp[i]=fp16(fp16(a[i]*cs)+fp16(rot*sn));
    }
    sync_grid();
    for(int i=rank();i<count*n;i+=threads()) a[i]=tmp[i];
    sync_grid();
}
__device__ void forward(Model m,Scratch s,int count,int pos) {
    Config c=m.c;
    int h=c.h,dim=h/c.heads,kv=c.kvheads*dim;
    for(int i=rank();i<count*h;i+=threads()) s.x[i]=__half2float(m.embed[(size_t)s.input[i/h]*h+i%h]);
    sync_grid();
    for(unsigned l=0;l<c.layers;l++) {
        Layer w=m.layer[l];
        rms(s.norm,s.x,w.norm,h,c.eps,count);
        matvec(s.q,w.q,w.qb,s.norm,h,h,count);
        matvec(s.k,w.k,w.kb,s.norm,kv,h,count);
        matvec(s.v,w.v,w.vb,s.norm,kv,h,count);
        rope(s.q,s.out,h,dim,pos,m.inverse,count);
        rope(s.k,s.norm,kv,dim,pos,m.inverse,count);
        float *keys=s.keys+(size_t)l*CTX*kv;
        float *vals=s.values+(size_t)l*CTX*kv;
        for(int i=rank();i<count*kv;i+=threads()) { keys[pos*kv+i]=s.k[i];vals[pos*kv+i]=s.v[i]; }
        sync_grid();
        int lane=threadIdx.x%32,warp=rank()/32;
        // Each warp owns one token/head and attends only through its position.
        for(unsigned item=warp;item<count*c.heads;item+=threads()/32) {
            int head=item%c.heads,tok=item/c.heads,kh=head/(c.heads/c.kvheads),end=pos+tok;
            float *prob=s.prob+item*CTX;
            for(int t=0;t<=end;t++) {
                float dot=0;
                for(int d=lane;d<dim;d+=32) dot+=s.q[tok*h+head*dim+d]*keys[t*kv+kh*dim+d];
                dot=warp_sum(dot);
                if(!lane) prob[t]=fp16(fp16(dot)/sqrtf((float)dim));
            }
            __syncwarp();
            if(!lane) {
                float mx=-INFINITY,den=0;
                for(int t=0;t<=end;t++) mx=fmaxf(mx,prob[t]);
                for(int t=0;t<=end;t++) den+=expf(prob[t]-mx);
                for(int t=0;t<=end;t++) prob[t]=fp16(expf(prob[t]-mx)/den);
            }
            __syncwarp();
            for(int d=lane;d<dim;d+=32) {
                float sum=0;
                for(int t=0;t<=end;t++) sum=fmaf(prob[t],vals[t*kv+kh*dim+d],sum);
                s.att[tok*h+head*dim+d]=fp16(sum);
            }
        }
        sync_grid();
        matvec(s.out,w.o,nullptr,s.att,h,h,count);
        for(int i=rank();i<count*h;i+=threads()) s.x[i]=fp16(s.x[i]+s.out[i]);
        sync_grid();
        rms(s.norm,s.x,w.post,h,c.eps,count);
        matvec(s.gate,w.gate,nullptr,s.norm,c.inter,h,count);
        matvec(s.up,w.up,nullptr,s.norm,c.inter,h,count);
        for(unsigned i=rank();i<count*c.inter;i+=threads()) s.gate[i]=fp16(fp16(s.gate[i]/(1+expf(-s.gate[i])))*s.up[i]);
        sync_grid();
        matvec(s.out,w.down,nullptr,s.gate,h,c.inter,count);
        for(int i=rank();i<count*h;i+=threads()) s.x[i]=fp16(s.x[i]+s.out[i]);
        sync_grid();
    }
}
__device__ unsigned sample(Model m, Scratch s, unsigned count) {
    __shared__ float maxima[256];
    __shared__ unsigned ids[256];
    rms(s.norm,s.x+(count-1)*m.c.h,m.norm,m.c.h,m.c.eps);
    matvec(s.logits,m.embed,nullptr,s.norm,m.c.vocab,m.c.h);
    if (blockIdx.x==0) {
    float mx=-INFINITY; unsigned id=0;
    for(unsigned i=threadIdx.x;i<m.c.vocab;i+=blockDim.x)
        if(s.logits[i]>mx || (s.logits[i]==mx && i<id)) { mx=s.logits[i]; id=i; }
    maxima[threadIdx.x]=mx; ids[threadIdx.x]=id;
    __syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {
            unsigned j=threadIdx.x+d;
            if(maxima[j]>maxima[threadIdx.x] || (maxima[j]==maxima[threadIdx.x] && ids[j]<ids[threadIdx.x])) {
                maxima[threadIdx.x]=maxima[j]; ids[threadIdx.x]=ids[j];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) s.control[4]=ids[0];
    }
    sync_grid();
    return s.control[4];
}
__global__ void resident(bgs_page *p, Model m, Scratch s, unsigned seen, bool once) {
    unsigned head, state, prompt, gen, token;
    for(;;) {
        sync_grid();
        if(!rank()) {
            s.control[0]=__ldcg(&p->llm_head);
            s.control[1]=__ldcg(&p->llm[seen%BGS_LLM_SLOTS].state);
            if(__ldcg((unsigned long long *)&p->stop_ns)) s.control[0]=0xffffffff;
        }
        sync_grid();
        head=s.control[0]; state=s.control[1];
        if(head==0xffffffff) return;
        if(head==seen || state!=BGS_PENDING) continue;
        bgs_llm_slot *slot=&p->llm[seen%BGS_LLM_SLOTS];
        if(!rank()) {
            unsigned np=__ldcg(&slot->n_prompt), ng=__ldcg(&slot->n_gen);
            bool bad=!np || np>BGS_LLM_MAX_TOK || !ng || ng>BGS_LLM_MAX_TOK;
            for(unsigned i=0;i<np && i<BGS_LLM_MAX_TOK;i++)
                if(__ldcg(&slot->tok_in[i])>=m.c.vocab) bad=true;
            s.control[2]=bad ? 0 : np; s.control[3]=ng;
            unsigned short flags=__ldcg(&slot->pad);
            __threadfence_system();
            __stcg(&slot->pad,(unsigned short)((flags & ~BGS_LLM_VALIDATING) |
                                             (bad ? BGS_LLM_REJECTED : 0)));
        }
        sync_grid();
        prompt=s.control[2]; gen=s.control[3];
        if(!prompt) {
            if(!rank()) {
                __stcg(&slot->produced,0u);
                __threadfence_system();
                __stcg(&slot->state,(unsigned)BGS_DONE);
            }
            sync_grid();
            seen++;
            if (once) return;
            continue;
        }
        for(unsigned i=rank();i<prompt;i+=threads()) s.input[i]=__ldcg(&slot->tok_in[i]);
        sync_grid();
        forward(m,s,prompt,0);
        for(unsigned k=0;k<gen;k++) {
            unsigned next=sample(m,s,k ? 1 : prompt);
            token=next;
            if(!rank()) {
                __stcg(&slot->tok_out[k],token);
                __threadfence_system();
                __stcg(&slot->produced,k+1);
            }
            sync_grid();
            if(k+1<gen) {
                if(!rank()) s.input[0]=token;
                sync_grid();
                forward(m,s,1,prompt+k);
            }
        }
        if(!rank()) {
            __threadfence_system();
            __stcg(&slot->state,(unsigned)BGS_DONE);
        }
        sync_grid();
        seen++;
        if (once) return;
    }
}

static volatile sig_atomic_t stopped;
static void stop_signal(int) { stopped=1; }

int main(int argc,char **argv) {
    signal(SIGTERM,stop_signal); signal(SIGINT,stop_signal);
    if(argc<2) { fprintf(stderr,"usage: qwen WEIGHTS.bin [seconds]\n"); return 1; }
    FILE *f=fopen(argv[1],"rb"); if(!f) { perror("weights"); return 1; }
    Config c; if(fread(&c,sizeof(c),1,f)!=1) return 1;
    if(c.h%32 || c.h%c.heads || c.heads%c.kvheads || (c.h/c.heads)%2) return 1;
    fseek(f,0,SEEK_END); size_t bytes=ftell(f)-sizeof(c); fseek(f,sizeof(c),SEEK_SET);
    std::vector<half> host(bytes/sizeof(half));
    if(fread(host.data(),1,bytes,f)!=bytes) return 1;
    fclose(f);
    half *weights; CUDA(cudaMalloc(&weights,bytes));
    CUDA(cudaMemcpy(weights,host.data(),bytes,cudaMemcpyHostToDevice));
    host.clear(); host.shrink_to_fit();
    half *cursor=weights;
    auto take=[&](size_t n) { half *p=cursor; cursor+=n; return p; };
    Model m={}; m.c=c; m.embed=take((size_t)c.vocab*c.h);
    int kv=c.kvheads*(c.h/c.heads);
    std::vector<Layer> layers(c.layers);
    for(auto &l:layers) {
        l.norm=take(c.h); l.q=take((size_t)c.h*c.h); l.qb=take(c.h);
        l.k=take((size_t)kv*c.h); l.kb=take(kv); l.v=take((size_t)kv*c.h); l.vb=take(kv);
        l.o=take((size_t)c.h*c.h); l.post=take(c.h); l.gate=take((size_t)c.inter*c.h);
        l.up=take((size_t)c.inter*c.h); l.down=take((size_t)c.h*c.inter);
    }
    m.norm=take(c.h);
    if((size_t)(cursor-weights)*sizeof(half)!=bytes) { fprintf(stderr,"weight layout mismatch\n"); return 1; }
    CUDA(cudaMalloc(&m.layer,c.layers*sizeof(Layer)));
    CUDA(cudaMemcpy(m.layer,layers.data(),c.layers*sizeof(Layer),cudaMemcpyHostToDevice));
    // Match Qwen2's CPU initialization: reciprocal of positive powers,
    // once during model loading. The GPU consumes the same fp32 frequencies.
    int dim=c.h/c.heads;
    std::vector<float> inverse(dim/2);
    for (int i=0;i<dim/2;i++) inverse[i]=1.0f/powf(c.theta,2.0f*i/dim);
    CUDA(cudaMalloc(&m.inverse,inverse.size()*sizeof(float)));
    CUDA(cudaMemcpy(m.inverse,inverse.data(),inverse.size()*sizeof(float),cudaMemcpyHostToDevice));
    Scratch s={};
    auto alloc=[&](float **p,size_t n) { CUDA(cudaMalloc(p,n*sizeof(float))); };
    alloc(&s.x,c.h*BGS_LLM_MAX_TOK); alloc(&s.norm,c.h*BGS_LLM_MAX_TOK); alloc(&s.q,c.h*BGS_LLM_MAX_TOK);
    alloc(&s.k,kv*BGS_LLM_MAX_TOK); alloc(&s.v,kv*BGS_LLM_MAX_TOK);
    alloc(&s.att,c.h*BGS_LLM_MAX_TOK); alloc(&s.out,c.h*BGS_LLM_MAX_TOK);
    alloc(&s.gate,c.inter*BGS_LLM_MAX_TOK); alloc(&s.up,c.inter*BGS_LLM_MAX_TOK);
    alloc(&s.prob,c.heads*CTX*BGS_LLM_MAX_TOK); alloc(&s.logits,c.vocab);
    CUDA(cudaMalloc(&s.input,BGS_LLM_MAX_TOK*sizeof(unsigned)));
    alloc(&s.keys,(size_t)c.layers*CTX*kv); alloc(&s.values,(size_t)c.layers*CTX*kv);
    CUDA(cudaMalloc(&s.control,5*sizeof(unsigned)));
    cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop,0));
    if (!prop.cooperativeLaunch) { fprintf(stderr,"cooperative grid unavailable\n"); return 1; }
    int blocks_per_sm; CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,resident,256,0));
    if (!blocks_per_sm) return 1;
    unsigned blocks=prop.multiProcessorCount;
    int fd=bpf_obj_get("/sys/fs/bpf/bgs_ctl"); if(fd<0) { perror("bpf_obj_get"); return 1; }
    auto *page=(bgs_page *)mmap(nullptr,BGS_PAGE_MMAP_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(page==MAP_FAILED) { perror("mmap"); return 1; }
    CUDA(cudaHostRegister(page,BGS_PAGE_MMAP_BYTES,cudaHostRegisterMapped));
    bgs_page *device; CUDA(cudaHostGetDevicePointer((void **)&device,page,0));
    int listener=socket(AF_INET,SOCK_STREAM,0), yes=1;
    setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in addr={}; addr.sin_family=AF_INET; addr.sin_port=htons(BGS_LLM_TCP_PORT);
    addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(bind(listener,(sockaddr *)&addr,sizeof(addr)) || listen(listener,SOMAXCONN)) { perror("listen"); return 1; }
    // Listener creation is bootstrap only. The module owns accept and RX drain.
    __atomic_store_n(&page->stop_ns,0,__ATOMIC_RELEASE);
    unsigned seen=0, launches=0;
#ifdef BGS_HOST_LAUNCH
    bool once=true;
    const char *dispatch="host-launch";
#else
    bool once=false;
    const char *dispatch="resident";
#endif
    // Both binaries launch this same kernel and use the same model math/page.
    void *args[]={&device,&m,&s,&seen,&once};
    if (!once) {
        CUDA(cudaLaunchCooperativeKernel((void *)resident,blocks,256,args));
        launches++;
    }
    int module=open("module/bpf_gpu_server_tx.ko",O_RDONLY|O_CLOEXEC);
    if(module<0) { perror("module open"); return 1; }
    char params[80]; snprintf(params,sizeof(params),"map_fd=%d listen_fd=%d",fd,listener);
    if(syscall(SYS_finit_module,module,params,0)) { perror("finit_module"); return 1; }
    close(module); close(fd);
    printf("resident Qwen ready: %u layers h=%u vocab=%u; %u CTAs, dispatch=%s\n",c.layers,c.h,c.vocab,blocks,dispatch);
    fflush(stdout);
    unsigned seconds=argc>2 ? atoi(argv[2]) : 60;
    if (once) {
        signal(SIGALRM,stop_signal); alarm(seconds);
        // Busy polling is the latency-oriented host-dispatch control. There
        // is no added sleep, copy, network worker or per-token host launch.
        while (!stopped) {
            if (__atomic_load_n(&page->llm_head,__ATOMIC_ACQUIRE)==seen ||
                __atomic_load_n(&page->llm[seen%BGS_LLM_SLOTS].state,__ATOMIC_ACQUIRE)!=BGS_PENDING)
                continue;
            CUDA(cudaLaunchCooperativeKernel((void *)resident,blocks,256,args));
            launches++;
            CUDA(cudaDeviceSynchronize());
            seen++;
        }
        alarm(0);
    } else {
        sleep(seconds);
    }
    __atomic_store_n(&page->stop_ns,1,__ATOMIC_RELEASE);
    cudaError_t finished=cudaDeviceSynchronize();
    if(syscall(SYS_delete_module,"bpf_gpu_server_tx",0)) { perror("delete_module"); return 1; }
    close(listener);
    CUDA(finished);
    CUDA(cudaHostUnregister(page)); munmap(page,BGS_PAGE_MMAP_BYTES);
    printf("resident Qwen stopped: dispatch=%s launches=%u\n",dispatch,launches);
    return 0;
}
