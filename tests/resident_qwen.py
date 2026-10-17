#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resident CUDA Qwen vs eager HF fp16 greedy oracle on real TCP replies."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier
import argparse
import ctypes
import signal
import gc
import os
import re
import socket
import struct
import subprocess
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def oracle_cases(gen, prefill_edges=False, name="Qwen/Qwen2.5-0.5B-Instruct"):
    tok=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16,attn_implementation="eager").to("cuda").eval()
    vocab=model.config.vocab_size
    print(f"HF reference model={name} type={model.config.model_type} vocab={vocab}",flush=True)
    cases=[]
    inputs=[tok(prompt).input_ids for prompt in ("The capital of France is", "What is 2 plus 2?", "Write a short greeting.", "Linux is")]
    if prefill_edges:
        pattern=inputs[0]
        inputs.extend((pattern*((n+len(pattern)-1)//len(pattern)))[:n] for n in (1,15,16,17,32,64))
    for ids in inputs:
        with torch.no_grad():
            out=model(torch.tensor([ids],device="cuda"),use_cache=True)
            past=out.past_key_values
            nxt=int(out.logits[:,-1].argmax(-1))
            expected=[nxt]
            for _ in range(gen-1):
                out=model(torch.tensor([[nxt]],device="cuda"),past_key_values=past,use_cache=True)
                past=out.past_key_values
                nxt=int(out.logits[:,-1].argmax(-1))
                expected.append(nxt)
        cases.append((ids,expected))
    del model,out,past
    gc.collect(); torch.cuda.empty_cache()
    return cases,vocab


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gen",type=int,default=8,choices=range(1,65))
    parser.add_argument("--executor",default="./build/qwen")
    parser.add_argument("--prefill-edges",action="store_true",help="also test 1/15/16/17/32/64-token prompt lengths")
    parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct",help="HF checkpoint; other models require a graph executor")
    args=parser.parse_args()
    server=(os.environ.get("BGS_BIND","127.0.0.1"),39403)
    user_network=Path(args.executor).name in ("qwen_graph_user_network","qwen_graph_user_network_wait")
    if args.model!="Qwen/Qwen2.5-0.5B-Instruct" and Path(args.executor).name not in ("qwen_graph","qwen_graph_host_launch","qwen_graph_user_network","qwen_graph_user_network_wait"):
        parser.error("--model requires qwen_graph or qwen_graph_host_launch")
    weights="/workspaces/.cache/bpf-gpu-server/qwen25-05b-fp16.bin" if args.model=="Qwen/Qwen2.5-0.5B-Instruct" else args.model
    cases,vocab=oracle_cases(args.gen,args.prefill_edges,args.model)
    valid=305+3*len(cases)
    print(f"HF eager fp16 oracle ready, model={args.model}, model released",flush=True)
    subprocess.run(["./build/bpf_gpu_server_load","detach","lo"],check=True)
    subprocess.run(["./build/bpf_gpu_server_load","stream-detach"],check=True)
    if not user_network:
        subprocess.run(["./build/bpf_gpu_server_load","stream-attach"],check=True)
    else:
        assert not Path("/sys/module/bpf_gpu_server_tx").exists(),"kernel TX must be absent for userspace control"
    executor=None
    complete=False
    logpath="/tmp/bpf-gpu-server-resident-qwen.log"
    try:
        with open(logpath,"w") as log:
            executor=subprocess.Popen(["strace","-D","-f","-e","trace=network,write","-o","/tmp/bpf-gpu-server-qwen-network.trace",args.executor,weights,"300"],stdout=log,stderr=subprocess.STDOUT)
        while "resident Qwen ready" not in open(logpath).read():
            if executor.poll() is not None:
                raise AssertionError(open(logpath).read())
            time.sleep(.1)
        for _ in range(5000):
            socket.create_connection(server,timeout=5).close()
        owner="userspace" if user_network else "kernel"
        print(f"PASS 5000 empty connections accepted/drained by {owner}; exceeds undrained listener backlog",flush=True)
        def request(idx, ids, expected, http=False, client=None, split=True, reject=False):
            with (nullcontext(client) if client is not None else socket.create_connection(server,timeout=60)) as client:
                start=time.perf_counter()
                client.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                wire=struct.pack("<IIIIQ",0x514c4d51,len(ids),len(expected),0,time.monotonic_ns())+struct.pack(f"<{len(ids)}I",*ids)
                if http:
                    wire=(f"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(wire)}\r\n\r\n".encode()+wire)
                # Separate writes across header and payload; the parser
                # must frame one request after TCP stream reassembly.
                if split:
                    for byte in wire:
                        client.sendall(bytes([byte]))
                        time.sleep(.001)
                else:
                    client.sendall(wire)
                reply=[] if reject else expected
                if http:
                    header=b""
                    while not header.endswith(b"\r\n\r\n"):
                        chunk=client.recv(1); assert chunk,"short HTTP header"; header+=chunk
                    status=b"400 Bad Request" if reject else b"200 OK"
                    assert header.startswith(b"HTTP/1.1 "+status+b"\r\n"),header
                    assert f"Content-Length: {len(reply)*4}\r\n".encode() in header,header
                data=b""
                while len(data)<len(reply)*4:
                    chunk=client.recv(len(reply)*4-len(data))
                    assert chunk,"short reply"
                    data+=chunk
                actual=list(struct.unpack(f"<{len(reply)}I",data))
                elapsed=time.perf_counter()-start
                if split:
                    print(f"http={http} case={idx} expected={expected} actual={actual} seconds={elapsed:.3f}",flush=True)
                assert actual==reply,(idx,actual,reply)
        for repetition in range(3):
            for idx,(ids,expected) in enumerate(cases):
                request(idx,ids,expected,http=repetition==1)
        with ThreadPoolExecutor(max_workers=8) as clients:
            futures=[clients.submit(request,i,*cases[i%len(cases)],i>=4) for i in range(8)]
            for future in futures: future.result()
        print("PASS eight concurrent clients, exact model replies, no admission drops",flush=True)
        # A reply must return queue ownership before the immediate next send.
        # Eight persistent closed-loop clients repeatedly wrap the eight slots.
        barrier=Barrier(8)
        def closed_loop(i):
            with socket.create_connection(server,timeout=60) as client:
                barrier.wait(timeout=60)
                for repeat in range(32):
                    ids,expected=cases[(i+repeat)%len(cases)]
                    invalid=ids.copy(); invalid[0]=vocab
                    request(i+8*repeat,invalid,expected,http=True,client=client,split=False,reject=True)
                    request(i+8*repeat,ids,expected,http=True,client=client,split=False)
        with ThreadPoolExecutor(max_workers=8) as clients:
            futures=[clients.submit(closed_loop,i) for i in range(8)]
            for future in futures: future.result()
        print("PASS 256 HTTP rejections immediately followed by 256 exact valid replies on eight persistent clients, repeated ring wrap",flush=True)
        # Different decode lengths complete out of admission order. Immediate
        # reuse must preserve the head-indexed ring and independent KV owners.
        mixed_barrier=Barrier(8)
        def mixed_loop(i):
            with socket.create_connection(server,timeout=60) as client:
                mixed_barrier.wait(timeout=60)
                for repeat in range(4):
                    ids,expected=cases[(i+repeat)%len(cases)]
                    count=1 if (i+repeat)%2 else args.gen
                    request(i+8*repeat,ids,expected[:count],http=True,client=client,split=False)
        with ThreadPoolExecutor(max_workers=8) as clients:
            futures=[clients.submit(mixed_loop,i) for i in range(8)]
            for future in futures: future.result()
        print("PASS 32 mixed-length exact replies on eight persistent clients, immediate reuse and ring wrap",flush=True)
        # Pipelining on one persistent stream crosses the slot ring boundary.
        with socket.create_connection(server,timeout=60) as client:
            for i in range(6):
                ids,expected=cases[i%len(cases)]
                body=struct.pack("<IIIIQ",0x514c4d51,len(ids),len(expected),0,time.monotonic_ns())+struct.pack(f"<{len(ids)}I",*ids)
                client.sendall(f"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body)
            client.shutdown(socket.SHUT_WR)
            for i in range(6):
                _,expected=cases[i%len(cases)]
                header=b""
                while not header.endswith(b"\r\n\r\n"):
                    chunk=client.recv(1); assert chunk,"short pipelined HTTP header"; header+=chunk
                assert header.startswith(b"HTTP/1.1 200 OK\r\n"),header
                assert f"Content-Length: {len(expected)*4}\r\n".encode() in header,header
                data=b""
                while len(data)<len(expected)*4:
                    chunk=client.recv(len(expected)*4-len(data)); assert chunk; data+=chunk
                actual=list(struct.unpack(f"<{len(expected)}I",data))
                assert actual==expected,("pipeline",i,actual,expected)
        print("PASS six pipelined HTTP responses in order, ring wrap, half-close",flush=True)
        ids,expected=cases[0]
        bad_last=ids.copy(); bad_last[-1]=vocab
        bad_first=ids.copy(); bad_first[0]=0xffffffff
        def body(tokens):
            return struct.pack("<IIIIQ",0x514c4d51,len(tokens),len(expected),0,time.monotonic_ns())+struct.pack(f"<{len(tokens)}I",*tokens)
        # An invalid request must not terminate the grid or poison a later
        # request on this same stream, including coalesced skb framing.
        with socket.create_connection(server,timeout=60) as client:
            for tokens in (bad_last,ids,bad_first,ids):
                payload=body(tokens)
                client.sendall(f"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode()+payload)
            for invalid in (True,False,True,False):
                header=b""
                while not header.endswith(b"\r\n\r\n"):
                    chunk=client.recv(1); assert chunk,"short rejection/pipeline header"; header+=chunk
                status=b"400 Bad Request" if invalid else b"200 OK"
                assert header.startswith(b"HTTP/1.1 "+status+b"\r\n"),header
                count=0 if invalid else len(expected)*4
                assert f"Content-Length: {count}\r\n".encode() in header,header
                data=b""
                while len(data)<count:
                    chunk=client.recv(count-len(data)); assert chunk; data+=chunk
                if not invalid: assert list(struct.unpack(f"<{len(expected)}I",data))==expected
        with socket.create_connection(server,timeout=60) as client:
            client.sendall(body(bad_first))
            assert client.recv(1)==b"","binary rejection must close the connection"
        request(0,ids,expected)
        print("PASS invalid first/last token IDs: HTTP 400, binary EOF, same-stream and later requests still exact",flush=True)
        network="userspace network control" if user_network else "no host accept/read/send worker"
        print(f"PASS {valid} valid TCP/HTTP requests, {valid*args.gen-16*(args.gen-1)} real model greedy tokens, 259 rejections, {network}",flush=True)
        complete=True
    finally:
        try:
            if executor is not None:
                shutdown=time.monotonic()
                if complete and Path(args.executor).name in ("qwen_graph","qwen_graph_host_launch","qwen_graph_user_network","qwen_graph_user_network_wait"):
                    # A signal on a CUDA helper must also wake the host wait.
                    tid=next(int(p.name) for p in Path(f"/proc/{executor.pid}/task").iterdir() if int(p.name)!=executor.pid)
                    assert ctypes.CDLL(None).tgkill(executor.pid,tid,signal.SIGTERM)==0
                    print(f"sent SIGTERM to executor helper tid={tid}",flush=True)
                else: executor.terminate()
                executor.wait()
                print(f"executor shutdown seconds={time.monotonic()-shutdown:.3f}",flush=True)
                log=open(logpath).read()
                print(log,flush=True)
                if complete and "dispatch=host-launch" in log:
                    assert f"dispatch=host-launch launches={valid+259}" in log,log
                elif complete:
                    assert "dispatch=resident launches=1" in log,log
                if complete and "device_graph_launches=" in log:
                    assert f"device_graph_launches={valid*args.gen-16*(args.gen-1)} graph_error=0 banks=8" in log,log
                    if args.gen==64: assert "max_active=8" in log,log
                    print("PASS concurrent graph-bank accounting and zero graph errors",flush=True)
                if complete: print("PASS expected CUDA launch count for the selected dispatch",flush=True)
                trace=open("/tmp/bpf-gpu-server-qwen-network.trace").read()
                if complete: assert "resident Qwen ready" in trace,"missing executor ready marker in trace"
                if "resident Qwen ready" in trace:
                    active=trace.split("resident Qwen ready",1)[1]
                    if user_network:
                        for syscall in ("accept4","recvfrom","sendmsg"):
                            assert re.search(r"\b"+syscall+r"\(",active),syscall
                        assert "userspace TCP stopped: accepted=5058 published=594 completed=594 rejected=259" in log,log
                        assert not Path("/sys/module/bpf_gpu_server_tx").exists()
                        print("PASS traced userspace accept/receive/send control with no kernel TX module",flush=True)
                    else:
                        assert not re.search(r"\b(?:accept4?|recvfrom|recvmsg|recvmmsg|sendto|sendmsg|sendmmsg)\(",active),active
                        print("PASS traced all executor threads: no accept/receive/send syscalls after ready",flush=True)
            if not user_network:
                subprocess.run(["./build/bpf_gpu_server_load","stats"],check=True)
        finally:
            subprocess.run(["./build/bpf_gpu_server_load","stream-detach"],check=True)
            subprocess.run(["./build/bpf_gpu_server_load","detach","lo"],check=True)
            for name in ("bgs_ctl","bgs_stats","bgs_db"):
                Path("/sys/fs/bpf",name).unlink(missing_ok=True)
            print("PASS test map pins removed after module shutdown and hook detach",flush=True)



if __name__=="__main__": main()
