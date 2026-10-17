// SPDX-License-Identifier: Apache-2.0
// Native remote load worker. Data layout is documented in graph_network_client.py.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sched.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>
#include "bpf_gpu_server_queue.h"
static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"log/wire use little endian");
struct Pattern { float input[64],expected[64]; };
#pragma pack(push,1)
struct Frame { uint32_t magic,np,ng,flags;uint64_t scheduled;float input[64]; };
struct Record {
    uint64_t index,scheduled,begin,sent,complete,end;
    uint32_t pattern,status,bytes,error;
    unsigned char reply[256];
};
struct Tick { uint64_t stamp,target,lag; };
#pragma pack(pop)
static_assert(sizeof(Frame)==280 && sizeof(Record)==320 && sizeof(Tick)==24,"format1");
static uint64_t ns(clockid_t clock=CLOCK_MONOTONIC) {
    timespec t{};if(clock_gettime(clock,&t))throw std::runtime_error("clock_gettime");
    return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
static void require(bool yes,const char* what) { if(!yes)throw std::runtime_error(what); }
template<class T> static std::vector<T> read_file(const std::string& name,size_t count) {
    FILE* f=fopen(name.c_str(),"rb");require(f,"open input");std::vector<T> out(count);
    require(fread(out.data(),sizeof(T),count,f)==count,"short input");
    require(fgetc(f)==EOF,"long input");fclose(f);return out;
}
static void write_file(const std::string& path,const std::string& text) {
    std::string tmp=path+".publishing";FILE* f=fopen(tmp.c_str(),"wx");
    require(f,"open metadata");require(fwrite(text.data(),1,text.size(),f)==text.size(),"metadata write");
    require(fclose(f)==0,"metadata close");require(rename(tmp.c_str(),path.c_str())==0,"metadata publish");
}
struct Conn {
    int fd=-1,phase=0; // 0 idle,1 connecting,2 sending,3 receiving
    uint64_t index=0;size_t written=0,received=0;
    Frame frame{};Record rec{};
};
struct Worker {
    std::string host,base,epoch_file;unsigned part,workers,connections;
    uint64_t count,epoch=0,done=0,failures=0,assigned=0;
    int ep=-1,timer=-1;FILE *records=nullptr,*ticks=nullptr;
    std::vector<uint64_t> offsets;std::vector<Pattern> patterns;std::vector<Conn> peers;
    static constexpr uint32_t TIMER=0xffffffffU;
    void watch(unsigned i,bool add=false) {
        auto& c=peers[i];if(c.fd<0)return;
        epoll_event e{};e.data.u32=i;e.events=EPOLLIN|EPOLLRDHUP;
        if(c.phase==1 || c.phase==2)e.events|=EPOLLOUT;
        require(epoll_ctl(ep,add?EPOLL_CTL_ADD:EPOLL_CTL_MOD,c.fd,&e)==0,"epoll watch");
    }
    void close_peer(Conn& c) {
        if(c.fd>=0){epoll_ctl(ep,EPOLL_CTL_DEL,c.fd,nullptr);close(c.fd);c.fd=-1;}
    }
    int socket_new() {
        int fd=socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(fd<0)return -1;int one=1;
        if(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))){close(fd);return -1;}
        return fd;
    }
    sockaddr_in address() {
        sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(BGS_LLM_TCP_PORT);
        require(inet_pton(AF_INET,host.c_str(),&a.sin_addr)==1,"numeric IPv4 required");return a;
    }
    void finish(unsigned i,unsigned status,unsigned error=0) {
        auto& c=peers[i];c.rec.status=status;c.rec.error=error;
        c.rec.bytes=c.received;c.rec.end=ns();
        require(fwrite(&c.rec,sizeof(c.rec),1,records)==1,"record write");
        done++;failures+=(status!=0);c.index+=uint64_t(workers)*connections;
        c.phase=0;c.written=0;c.received=0;
        if(status || c.index>=count)close_peer(c);else watch(i);
    }
    void send_ready(unsigned i) {
        auto& c=peers[i];
        while(c.phase==2 && c.written<sizeof(Frame)) {
            ssize_t n=send(c.fd,(unsigned char*)&c.frame+c.written,sizeof(Frame)-c.written,MSG_NOSIGNAL);
            if(n>0)c.written+=n;
            else if(n<0 && errno==EINTR)continue;
            else if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK)){watch(i);return;}
            else {unsigned error=errno;finish(i,5,error);return;}
        }
        if(c.phase==2){c.rec.sent=ns();c.phase=3;watch(i);}
    }
    void start(unsigned i) {
        uint64_t now=ns();auto& c=peers[i];c.rec={};c.rec.index=c.index;c.rec.pattern=c.index%16;
        c.rec.scheduled=epoch+offsets[c.index];c.rec.begin=now;c.written=c.received=0;
        if(now>=c.rec.scheduled+30000000000ULL){finish(i,1);return;}
        c.frame={};c.frame.magic=BGS_LLM_MAGIC;c.frame.np=c.frame.ng=64;
        c.frame.scheduled=c.rec.scheduled;
        memcpy(c.frame.input,patterns[c.rec.pattern].input,256);
        if(c.fd<0) {
            c.fd=socket_new();if(c.fd<0){finish(i,2,errno);return;}
            auto a=address();int code=connect(c.fd,(sockaddr*)&a,sizeof(a));
            if(code && errno!=EINPROGRESS){unsigned error=errno;close(c.fd);c.fd=-1;finish(i,2,error);return;}
            c.phase=code?1:2;watch(i,true);
        } else {c.phase=2;watch(i);}
        if(c.phase==2)send_ready(i);
    }
    void received(unsigned i) {
        auto& c=peers[i];
        while(c.phase==3 && c.received<256) {
            ssize_t n=recv(c.fd,c.rec.reply+c.received,256-c.received,0);
            if(n>0)c.received+=n;
            else if(n<0 && errno==EINTR)continue;
            else if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK))return;
            else {unsigned error=n?errno:0;finish(i,n?4:3,error);return;}
        }
        if(c.phase==3 && c.received==256) {
            c.rec.complete=ns();
            if(c.rec.complete>=c.rec.scheduled+30000000000ULL){finish(i,1);return;}
            bool valid=true;
            for(unsigned j=0;j<64;j++) {
                float got;memcpy(&got,c.rec.reply+4*j,4);
                if(got!=patterns[c.rec.pattern].expected[j]){valid=false;break;}
            }
            finish(i,valid?0:6);
        }
    }
    void setup(const std::string& schedule,const std::string& pattern_file) {
        offsets=read_file<uint64_t>(schedule,count);patterns=read_file<Pattern>(pattern_file,16);
        records=fopen((base+".records.bin").c_str(),"wbx");
        ticks=fopen((base+".timers.bin").c_str(),"wbx");require(records&&ticks,"open logs");
        ep=epoll_create1(EPOLL_CLOEXEC);timer=timerfd_create(CLOCK_MONOTONIC,TFD_NONBLOCK|TFD_CLOEXEC);
        require(ep>=0 && timer>=0,"epoll/timer setup");
        epoll_event e{};e.data.u32=TIMER;e.events=EPOLLIN;require(!epoll_ctl(ep,EPOLL_CTL_ADD,timer,&e),"timer watch");
        peers.resize(connections);auto a=address();std::string contexts="[";
        for(unsigned i=0;i<connections;i++) {
            auto& c=peers[i];c.index=part+uint64_t(workers)*i;
            if(c.index<count)assigned+=(count-1-c.index)/(uint64_t(workers)*connections)+1;
            c.fd=socket_new();require(c.fd>=0,"initial socket");
            int code=connect(c.fd,(sockaddr*)&a,sizeof(a));
            require(code==0 || errno==EINPROGRESS,"initial connect");
            if(code) {
                pollfd p{c.fd,POLLOUT,0};require(poll(&p,1,30000)>0,"initial connect timeout");
                int err=0;socklen_t len=sizeof(err);require(!getsockopt(c.fd,SOL_SOCKET,SO_ERROR,&err,&len)&&!err,"initial connect error");
            }
            sockaddr_in local{};socklen_t len=sizeof(local);require(!getsockname(c.fd,(sockaddr*)&local,&len),"sockname");
            if(i)contexts+=",";
            contexts+="{\"index\":"+std::to_string(i)+",\"local_port\":"+std::to_string(ntohs(local.sin_port))+",\"tcp_nodelay\":1}";
            watch(i,true);
        }
        contexts+="]";cpu_set_t mask;CPU_ZERO(&mask);require(!sched_getaffinity(0,sizeof(mask),&mask),"affinity");
        std::string cpus="[";bool comma=false;
        for(int i=0;i<CPU_SETSIZE;i++)if(CPU_ISSET(i,&mask)){if(comma)cpus+=",";cpus+=std::to_string(i);comma=true;}cpus+="]";
        write_file(base+".ready.json","{\"format\":1,\"part\":"+std::to_string(part)+",\"pid\":"+std::to_string(getpid())+
          ",\"ready_ns\":"+std::to_string(ns())+",\"cpu_affinity\":"+cpus+",\"connections\":"+contexts+"}");
    }
    void run() {
        for(;;) {
            std::ifstream f(epoch_file);if(f>>epoch)break;usleep(1000);
        }
        for(auto& c:peers)if(c.index>=count)close_peer(c);
        uint64_t next_tick=epoch+1000000;
        while(done<assigned) {
            uint64_t now=ns();
            while(next_tick<=now){Tick t{now,next_tick,now-next_tick};require(fwrite(&t,sizeof(t),1,ticks)==1,"timer log");next_tick+=1000000;}
            for(unsigned i=0;i<connections;i++) {
                auto& c=peers[i];if(c.index>=count)continue;
                if(c.phase && now>=c.rec.scheduled+30000000000ULL)finish(i,1);
                else if(!c.phase && epoch+offsets[c.index]<=now)start(i);
            }
            if(done==assigned)break;
            uint64_t next=next_tick;
            for(auto& c:peers)if(c.index<count)next=std::min<uint64_t>(next,c.phase?c.rec.scheduled+30000000000ULL:epoch+offsets[c.index]);
            now=ns();if(next<=now)next=now+1;
            itimerspec spec{};spec.it_value.tv_sec=next/1000000000ULL;spec.it_value.tv_nsec=next%1000000000ULL;
            require(!timerfd_settime(timer,TFD_TIMER_ABSTIME,&spec,nullptr),"arm timer");
            epoll_event events[128];int n=epoll_wait(ep,events,128,-1);
            if(n<0 && errno==EINTR)continue;require(n>=0,"epoll wait");
            for(int k=0;k<n;k++) {
                unsigned i=events[k].data.u32;
                if(i==TIMER){uint64_t value;ssize_t ignored=read(timer,&value,sizeof(value));(void)ignored;continue;}
                auto& c=peers[i];if(c.fd<0)continue;
                if(c.phase==1 && events[k].events&(EPOLLOUT|EPOLLERR|EPOLLHUP)) {
                    int error=0;socklen_t len=sizeof(error);
                    if(getsockopt(c.fd,SOL_SOCKET,SO_ERROR,&error,&len)||error){finish(i,2,error?error:errno);continue;}
                    c.phase=2;
                }
                if(c.phase==2 && events[k].events&EPOLLOUT)send_ready(i);
                if(c.phase==3 && events[k].events&(EPOLLIN|EPOLLRDHUP|EPOLLHUP|EPOLLERR))received(i);
                if(c.fd>=0 && !c.phase && events[k].events&(EPOLLRDHUP|EPOLLHUP|EPOLLERR))close_peer(c);
            }
        }
        for(auto& c:peers)close_peer(c);
        require(!fclose(records)&&!fclose(ticks),"close logs");close(timer);close(ep);
        write_file(base+".terminal.json","{\"part\":"+std::to_string(part)+",\"records\":"+std::to_string(done)+
          ",\"failures\":"+std::to_string(failures)+",\"end_ns\":"+std::to_string(ns())+
          ",\"cpu_ns\":"+std::to_string(ns(CLOCK_PROCESS_CPUTIME_ID))+"}");
    }
};
int main(int argc,char** argv) {
    try {
        require(argc==10,"usage: worker host part workers connections count schedule patterns epoch outputbase");
        Worker w;w.host=argv[1];w.part=std::stoul(argv[2]);w.workers=std::stoul(argv[3]);w.connections=std::stoul(argv[4]);
        w.count=std::stoull(argv[5]);w.epoch_file=argv[8];w.base=argv[9];
        require(w.workers && w.connections && w.part<w.workers,"invalid worker geometry");
        w.setup(argv[6],argv[7]);w.run();return 0;
    } catch(const std::exception& e){fprintf(stderr,"%s errno=%d\n",e.what(),errno);return 1;}
}
