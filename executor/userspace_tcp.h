// SPDX-License-Identifier: Apache-2.0
// Matched userspace network control: same page ABI and resident GPU dispatcher.
#include <sys/epoll.h>
#include <sys/uio.h>
#include <sys/eventfd.h>
#include <cerrno>
#include <cstring>
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <chrono>

static volatile sig_atomic_t user_wake_fd=-1;
static void user_network_interrupt() {
    int fd=user_wake_fd;
    if(fd>=0) { uint64_t one=1;ssize_t ignored=write(fd,&one,sizeof(one));(void)ignored; }
}

static int user_network(bgs_page *page,int listener,unsigned seconds,
                        volatile sig_atomic_t *stopped) {
    struct Peer {
        int fd;
        bool eof=false,queued=false;
        int slot=-1;
        std::string input;
    };
    struct Pending {
        std::shared_ptr<Peer> peer;
        size_t offset,bytes;
        unsigned prompt,gen;
        unsigned long long client_ns;
        bool http;
    };
    std::deque<Pending> pending;
    size_t queued_peak=0;
    struct Reply {
        std::shared_ptr<Peer> peer;
        std::string header;
        size_t header_sent=0,bytes=0;
    } replies[BGS_LLM_SLOTS];
    std::map<int,std::shared_ptr<Peer>> peers;
    unsigned head=0,completed=0,rejected_count=0,accepted=0;
    int ep=epoll_create1(EPOLL_CLOEXEC);
    if(ep<0) { perror("epoll_create1");return 1; }
    fcntl(listener,F_SETFL,fcntl(listener,F_GETFL)|O_NONBLOCK);
    epoll_event watch={};watch.events=EPOLLIN;watch.data.fd=listener;
    if(epoll_ctl(ep,EPOLL_CTL_ADD,listener,&watch)) { close(ep);return 1; }
    int wake=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
    if(wake<0) { close(ep);return 1; }
    watch.events=EPOLLIN;watch.data.fd=wake;
    if(epoll_ctl(ep,EPOLL_CTL_ADD,wake,&watch)) { close(wake);close(ep);return 1; }
    user_wake_fd=wake;
    auto disconnect=[&](Peer &peer) {
        if(peer.fd>=0) {
            epoll_ctl(ep,EPOLL_CTL_DEL,peer.fd,nullptr);
            close(peer.fd);peer.fd=-1;
        }
        peer.eof=true;
    };
    // Queue each complete eligible frame once, when readiness is observed.
    // A connection retains its existing one-in-flight response-order contract.
    auto enqueue=[&](const std::shared_ptr<Peer> &peer) {
        if(peer->fd<0 || peer->slot>=0 || peer->queued || peer->input.empty())return;
        size_t offset=0;
        bool http=peer->input.compare(0,4,"POST")==0;
        if(http) {
            size_t end=peer->input.find("\r\n\r\n");
            if(end==std::string::npos)return;
            offset=end+4;
        }
        if(peer->input.size()<offset+24)return;
        unsigned fields[4];unsigned long long client_ns;
        memcpy(fields,peer->input.data()+offset,16);
        memcpy(&client_ns,peer->input.data()+offset+16,8);
        unsigned prompt=fields[1],gen=fields[2];
        if(fields[0]!=BGS_LLM_MAGIC || !prompt || prompt>BGS_LLM_MAX_TOK ||
           !gen || gen>BGS_LLM_MAX_TOK) { disconnect(*peer);return; }
        size_t bytes=24+prompt*4;
        if(http) {
            size_t key=peer->input.find("\r\nContent-Length: ");
            if(key==std::string::npos || key>=offset ||
               strtoul(peer->input.c_str()+key+18,nullptr,10)!=bytes) {
                disconnect(*peer);return;
            }
        }
        if(peer->input.size()<offset+bytes)return;
        pending.push_back({peer,offset,bytes,prompt,gen,client_ns,http});
        peer->queued=true;
        if(pending.size()>queued_peak)queued_peak=pending.size();
    };
    auto send_reply=[&](Peer &peer,Reply &reply,bgs_llm_slot *slot,unsigned produced) {
        iovec vec[2];unsigned count=0;
        size_t header_bytes=reply.header.size()-reply.header_sent;
        if(header_bytes) vec[count++]={reply.header.data()+reply.header_sent,header_bytes};
        if(produced*4>reply.bytes)
            vec[count++]={(char *)slot->tok_out+reply.bytes,produced*4-reply.bytes};
        if(!count) return;
        msghdr msg={};msg.msg_iov=vec;msg.msg_iovlen=count;
        ssize_t n=sendmsg(peer.fd,&msg,MSG_DONTWAIT|MSG_NOSIGNAL);
        if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)
            disconnect(peer);
        if(n>0) {
            size_t sent_header=(size_t)n<header_bytes ? (size_t)n : header_bytes;
            reply.header_sent+=sent_header;reply.bytes+=(size_t)n-sent_header;
        }
    };
#ifdef BGS_USER_NETWORK_WAIT
    printf("userspace completion wait=epoll_pwait2 active_timeout_ns=60000 idle=socket_or_deadline shutdown=eventfd\n");
#else
    printf("userspace completion wait=busy epoll_wait timeout_ms=0\n");
#endif
    fflush(stdout);
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
    while(!*stopped && std::chrono::steady_clock::now()<deadline) {
        epoll_event events[BGS_LLM_SLOTS*2];
#ifdef BGS_USER_NETWORK_WAIT
        // Socket readiness wakes immediately; GPU completions are polled after
        // the same nominal 60 us lower wait bound used by kernel TX.
        auto remaining=std::chrono::duration_cast<std::chrono::nanoseconds>(
            deadline-std::chrono::steady_clock::now()).count();
        if(remaining<=0)break;
        long long wait_ns=(head!=completed || !pending.empty()) ? 60000 : remaining;
        const timespec timeout={wait_ns/1000000000,wait_ns%1000000000};
        int n=epoll_pwait2(ep,events,BGS_LLM_SLOTS*2,&timeout,nullptr);
#else
        int n=epoll_wait(ep,events,BGS_LLM_SLOTS*2,0);
#endif
        if(*stopped)break;
        if(n<0 && errno!=EINTR) { perror("epoll wait");break; }
        for(int e=0;e<n;e++) {
            int fd=events[e].data.fd;
            if(fd==wake) { uint64_t count;ssize_t ignored=read(wake,&count,sizeof(count));(void)ignored;continue; }
            if(fd==listener) {
                while(true) {
                    int child=accept4(listener,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
                    if(child<0) break;
                    auto peer=std::make_shared<Peer>();peer->fd=child;
                    peers[child]=peer;accepted++;
                    watch.events=EPOLLIN|EPOLLRDHUP;watch.data.fd=child;
                    if(epoll_ctl(ep,EPOLL_CTL_ADD,child,&watch)) disconnect(*peer);
                }
            } else {
                auto it=peers.find(fd);if(it==peers.end()) continue;
                Peer &peer=*it->second;
                char buffer[4096];
                while(peer.fd>=0) {
                    ssize_t got=recv(peer.fd,buffer,sizeof(buffer),MSG_DONTWAIT);
                    if(got>0) peer.input.append(buffer,got);
                    else {
                        if(got==0) {
                            peer.eof=true;epoll_ctl(ep,EPOLL_CTL_DEL,peer.fd,nullptr);
                        } else if(errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)
                            disconnect(peer);
                        break;
                    }
                }
                enqueue(it->second);
            }
        }
        for(unsigned i=0;i<BGS_LLM_SLOTS;i++) {
            Reply &reply=replies[i];if(!reply.peer) continue;
            auto peer_ref=reply.peer;Peer &peer=*peer_ref;bgs_llm_slot *slot=&page->llm[i];
            unsigned state=__atomic_load_n(&slot->state,__ATOMIC_ACQUIRE);
            unsigned short flags=__atomic_load_n(&slot->pad,__ATOMIC_ACQUIRE);
            bool rejected=flags&BGS_LLM_REJECTED;
            if((flags&BGS_LLM_VALIDATING) || (rejected && state!=BGS_DONE)) continue;
            unsigned produced=__atomic_load_n(&slot->produced,__ATOMIC_ACQUIRE);
            if(state!=BGS_DONE && produced==slot->n_gen && produced) produced--;
            if(rejected && (flags&BGS_LLM_TRANSPORT_MASK)==1) disconnect(peer);
            if((flags&BGS_LLM_TRANSPORT_MASK)==2 && reply.header.empty()) {
                reply.header="HTTP/1.1 "+std::string(rejected ? "400 Bad Request" : "200 OK")
                    +"\r\nContent-Type: application/octet-stream\r\nContent-Length: "
                    +std::to_string(rejected ? 0 : slot->n_gen*4)+"\r\n\r\n";
            }
            // Keep default TCP buffering; submit header and first tokens together.
            if(peer.fd>=0 && (produced || rejected))
                send_reply(peer,reply,slot,produced);
            if(state==BGS_DONE && (peer.fd<0 ||
                (reply.header_sent==reply.header.size() && reply.bytes==produced*4))) {
                peer.slot=-1;reply={};
                __atomic_store_n(&slot->state,(unsigned)BGS_FREE,__ATOMIC_RELEASE);
                __atomic_fetch_and(&page->llm_admission,~(1ULL<<(32+i)),__ATOMIC_RELEASE);
                completed++;if(rejected) rejected_count++;
                enqueue(peer_ref);
            }
        }
        for(auto it=peers.begin();it!=peers.end();) {
            auto peer=it->second;
            if(peer->slot<0 && (peer->fd<0 || (peer->eof && peer->input.empty()))) {
                disconnect(*peer);it=peers.erase(it);
            } else ++it;
        }
        while(!pending.empty()) {
            auto &request=pending.front();
            auto peer=request.peer;
            if(peer->fd<0) { peer->queued=false;pending.pop_front();continue; }
            unsigned i=head%BGS_LLM_SLOTS;bgs_llm_slot *slot=&page->llm[i];
            if(__atomic_load_n(&page->llm_reserved,__ATOMIC_ACQUIRE)&(1u<<i))break;
            __atomic_fetch_or(&page->llm_admission,1ULL<<(32+i),__ATOMIC_ACQ_REL);
            __atomic_store_n(&slot->state,(unsigned)BGS_WRITING,__ATOMIC_RELEASE);
            slot->n_prompt=request.prompt;slot->n_gen=request.gen;slot->produced=0;
            slot->client_ns=request.client_ns;
            slot->pad=(request.http ? 2 : 1)|BGS_LLM_VALIDATING;
            memcpy(slot->tok_in,peer->input.data()+request.offset+24,request.prompt*4);
            peer->input.erase(0,request.offset+request.bytes);
            replies[i].peer=peer;peer->slot=i;peer->queued=false;pending.pop_front();
            __atomic_store_n(&slot->state,(unsigned)BGS_PENDING,__ATOMIC_RELEASE);
            __atomic_store_n(&page->llm_head,++head,__ATOMIC_RELEASE);
        }
    }
    for(auto &item:peers) disconnect(*item.second);
    user_wake_fd=-1;close(wake);close(ep);
    printf("USER_ADMISSION policy=eligible_frame_fifo queued_peak=%zu queued_at_stop=%zu\n",queued_peak,pending.size());
    printf("userspace TCP stopped: accepted=%u published=%u completed=%u rejected=%u\n",
           accepted,head,completed,rejected_count);fflush(stdout);
    return 0;
}
