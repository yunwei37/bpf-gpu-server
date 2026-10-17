#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real vLLM graph TCP correctness and matched serving-policy comparison."""
import argparse,ctypes,json,math,os,re,signal,socket,statistics,struct,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from resident_dispatch import connect,request,worker_cpu_snapshot,worker_cpu_delta
from transformers import AutoConfig,AutoTokenizer

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True);ap.add_argument("--output",required=True)
    run_modes=ap.add_mutually_exclusive_group()
    run_modes.add_argument("--correctness",action="store_true")
    run_modes.add_argument("--preflight",action="store_true")
    ap.add_argument("--uring",action="store_true",help="Qualify only the stock io_uring transport")
    ap.add_argument("--untraced",action="store_true")
    ap.add_argument("--reference-path",type=Path)
    comparisons=ap.add_mutually_exclusive_group()
    comparisons.add_argument("--copy-comparison",action="store_true")
    comparisons.add_argument("--loop-comparison",action="store_true")
    comparisons.add_argument("--loop-progress-comparison",action="store_true")
    comparisons.add_argument("--transport-comparison",action="store_true")
    comparisons.add_argument("--packing-comparison",action="store_true")
    comparisons.add_argument("--release-comparison",action="store_true")
    comparisons.add_argument("--dispatch-comparison",action="store_true")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(),output
    raw=output.open("w",buffering=1);lock=threading.Lock()
    def emit(event,**kw):
        with lock:raw.write(json.dumps(dict(event=event,utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kw))+"\n")
    tok=AutoTokenizer.from_pretrained(args.model);config=AutoConfig.from_pretrained(args.model)
    prompts=[tok(text).input_ids for text in ("The capital of France is","What is 2 plus 2?","Write a short greeting.","Linux is")]
    modes=("separate-kernel","chained-kernel","separate-userspace","chained-userspace") if args.dispatch_comparison else ("serial-kernel","prefix-kernel","serial-userspace","prefix-userspace") if args.release_comparison else (("packed-kernel","packed-userspace") if args.correctness and args.packing_comparison else (
        ("grouped-kernel","packed-kernel","grouped-userspace","packed-userspace") if args.packing_comparison else ("kernel","userspace")))
    if args.uring:
        assert not (args.packing_comparison or args.release_comparison or args.dispatch_comparison or args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison)
        modes=("uring",)
    if args.transport_comparison:modes=("uring","userspace","kernel")
    if args.copy_comparison:modes=("uring","uring_full_copy","userspace")
    if args.loop_comparison:modes=("uring_loop","userspace","uring")
    if args.loop_progress_comparison:modes=("uring_loop","userspace","uring_loop_after")
    matrix=((64,8),) if args.copy_comparison else ((1,1),(64,1),(1,8),(64,8))
    repetitions=1 if args.correctness or args.preflight else (6 if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison else 5)
    requests_per_cell=8 if args.preflight else (256 if args.copy_comparison else 64)
    reference_model=None;reference_pool={}
    if args.untraced:
        assert args.reference_path is not None,"Untraced qualification requires pre-observation frozen complete references"
        asset=json.loads(args.reference_path.read_text())
        reference_model=next(m for m in asset["models"].values() if m["model"]==args.model)
        assert reference_model["vocab"]==config.vocab_size and asset["precision"]=="fp16"
        reference_pool={(tuple(r["prompt"]),r["gen"]):{tuple(s) for s in r["sequences"]} for r in reference_model["references"]}
        emit("frozen_complete_references",path=str(args.reference_path),source_commit=asset["source_commit"],model=reference_model,
             scope=asset["scope"])
    emit("metadata",model=args.model,vocab=config.vocab_size,prompts=prompts,precision="fp16",
         modes=modes,matrix=matrix,repetitions=repetitions,requests_per_cell=requests_per_cell,warmups_per_executor=8,
         correctness=args.correctness,preflight=args.preflight,untraced=args.untraced,comparison="kernel-loop-progress" if args.loop_progress_comparison else "kernel-loop" if args.loop_comparison else "stock-copy" if args.copy_comparison else "stock-transport" if args.transport_comparison else "dispatch" if args.dispatch_comparison else "release" if args.release_comparison else ("packing" if args.packing_comparison else "transport"),trace_condition="ordinary paths untraced; fixed complete-output-set fidelity" if args.untraced else "both transports identically BGS_TRACE",
         commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         driver_source=Path(__file__).read_text(),client_source=(root/"bench/resident_dispatch.py").read_text(),
         backend_source=(root/"executor/vllm_graph.py").read_text(),native_source=(root/"executor/vllm_graph.cu").read_text(),
         python=sys.version,torch=subprocess.check_output(["python3","-c","import torch;print(torch.__version__)"],text=True).strip())
    frozen={}
    if args.packing_comparison or args.release_comparison:
        for previous in (root/"bench/results/vllm_server_20261012").glob("*_correctness.jsonl"):
            previous_meta=next(json.loads(line) for line in previous.open() if json.loads(line)["event"]=="metadata")
            if previous_meta["model"]!=args.model:continue
            reference=previous.with_name(previous.stem+"_p0_kernel.gpu.jsonl.references.jsonl")
            first=[json.loads(line) for line in reference.open()][:10]
            frozen={tuple(row["prompt"]):row["tokens"] for row in first}
            emit("frozen_unpadded_reference",path=str(reference),references=first)
    exit_code=1
    try:
        for pair in range(repetitions):
            order=modes if len(modes)==2 or (pair//2)%2==0 else modes[2:]+modes[:2]
            if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison:order=tuple(permutations(modes))[pair%6]
            for mode in (order if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison or pair%2==0 else order[::-1]):
                kernel_mode=mode.endswith("kernel") or mode.startswith("uring")
                assert not Path("/sys/module/bpf_gpu_server_tx").exists()
                assert not Path("/sys/fs/bpf/bgs_ctl").exists()
                p=None;attached=False;rows=[];success=False
                prefix=output.with_suffix("").with_name(output.stem+f"_p{pair}_{mode}")
                trace=Path(str(prefix)+".gpu.jsonl");logpath=Path(str(prefix)+".stdout.txt")
                nettrace=Path(str(prefix)+".network.txt")
                budget=(3*10+sum(range(1,9))+8*32*2+8*4+6+4+2) if args.correctness else requests_per_cell*len(matrix)+8
                binary=(("vllm_graph_chained" if mode.startswith("chained") else "vllm_graph") if args.dispatch_comparison else ("vllm_graph" if mode.startswith("prefix") else "vllm_graph_padded") if args.release_comparison
                        else ("vllm_graph_padded" if mode.startswith("packed") else "vllm_graph_grouped") if args.packing_comparison
                        else "vllm_graph")+("" if kernel_mode else "_user_network_wait")+("" if args.untraced else "_trace")
                if mode.startswith("uring"):binary="vllm_graph_"+mode+("" if args.untraced else "_trace")
                command=["./build/"+binary,args.model,"300"]
                if not args.untraced:command+=["--trace-output",str(trace),"--trace-requests",str(budget)]
                if args.correctness:command=["strace","-D","-f","-e","trace=network,write,io_uring_enter","-o",str(nettrace),*command]
                try:
                    if kernel_mode:
                        r=subprocess.run(["./build/bpf_gpu_server_load","stream-uring-attach" if mode.startswith("uring") else "stream-attach"],capture_output=True,text=True)
                        emit("attach",pair=pair,mode=mode,code=r.returncode,stdout=r.stdout,stderr=r.stderr)
                        assert r.returncode==0;attached=True
                    with logpath.open("w") as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    emit("start",pair=pair,mode=mode,pid=p.pid,command=command,trace=str(trace))
                    print(f"START pair={pair} mode={mode} pid={p.pid}",flush=True)
                    while "resident vLLM ready" not in logpath.read_text():
                        if p.poll() is not None:raise RuntimeError(logpath.read_text()[-6000:])
                        time.sleep(.1)
                    emit("ready",pair=pair,mode=mode,pid=p.pid)
                    if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison:
                        tasks=[dict(tid=int(t.name),comm=(t/"comm").read_text().strip()) for t in Path(f"/proc/{p.pid}/task").iterdir()]
                        if mode.startswith("uring_loop"):assert not any(t["comm"].startswith("iou-sqp") for t in tasks),tasks
                        elif mode.startswith("uring"):assert any(t["comm"].startswith("iou-sqp") for t in tasks),tasks
                        emit("worker_tasks",pair=pair,mode=mode,pid=p.pid,tasks=tasks)
                    def keep(row,kind):
                        row=dict(row,kind=kind);rows.append(row)
                        emit("request",pair=pair,mode=mode,**row)
                        return row
                    def wire(ids,gen,http=True):
                        marker=time.monotonic_ns()
                        payload=struct.pack("<IIIIQ",0x514c4d51,len(ids),gen,0,marker)+struct.pack(f"<{len(ids)}I",*ids)
                        body=(f"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode()+payload) if http else payload
                        return marker,body
                    def receive(client,ids,gen,marker,http=True,bad=False):
                        head=b""
                        if http:
                            while not head.endswith(b"\r\n\r\n"):
                                chunk=client.recv(1);assert chunk,"short HTTP header";head+=chunk
                            status=b"400 Bad Request" if bad else b"200 OK"
                            assert head.startswith(b"HTTP/1.1 "+status+b"\r\n"),head
                            count=0 if bad else gen*4
                            assert f"Content-Length: {count}\r\n".encode() in head,head
                        elif bad:
                            assert client.recv(1)==b"","binary rejection must close"
                            count=0
                        else:count=gen*4
                        data=b""
                        while len(data)<count:
                            chunk=client.recv(count-len(data));assert chunk,"short body";data+=chunk
                        tokens=list(struct.unpack(f"<{count//4}I",data))
                        return dict(client_ns=marker,input_ids=ids,gen=gen,bad=bad,tokens=tokens,http=http)
                    def exchange(ids,gen,kind,http=True,split=False,bad=False,client=None):
                        own=client is None
                        if own:client=connect()
                        try:
                            marker,data=wire(ids,gen,http)
                            emit("request_sent",pair=pair,mode=mode,kind=kind,client_ns=marker,input_ids=ids,gen=gen,bad=bad,http=http)
                            if split:
                                for byte in data:client.sendall(bytes([byte]));time.sleep(.001)
                            else:client.sendall(data)
                            return keep(receive(client,ids,gen,marker,http,bad),kind)
                        finally:
                            if own:client.close()
                    if args.correctness:
                        for _ in range(5000):connect().close()
                        emit("empty_connections",pair=pair,mode=mode,count=5000)
                        cases=prompts+[(prompts[0]*math.ceil(n/len(prompts[0])))[:n] for n in (1,15,16,17,32,64)]
                        for repeat in range(3):
                            for ids in cases:exchange(ids,64,"fragmented",http=repeat==1,split=True)
                        def concurrent(count,iterations,kind,mixed=False,reject=False):
                            barrier=threading.Barrier(count)
                            def lane(i):
                                with connect() as client:
                                    barrier.wait(timeout=60)
                                    for repeat in range(iterations):
                                        ids=cases[(i+repeat)%len(cases)]
                                        gen=1 if mixed and (i+repeat)%2 else 64
                                        if reject:
                                            invalid=ids.copy();invalid[0]=config.vocab_size
                                            exchange(invalid,gen,kind+"_reject",bad=True,client=client)
                                        exchange(ids,gen,kind,client=client)
                            with ThreadPoolExecutor(max_workers=count) as pool:
                                futures=[pool.submit(lane,i) for i in range(count)]
                                for future in futures:future.result()
                        # Every client concurrency count is exercised; report actual GPU batches.
                        for count in range(1,9):concurrent(count,1,"concurrency_"+str(count))
                        concurrent(8,32,"wrap",reject=True)
                        concurrent(8,4,"mixed",mixed=True)
                        with connect() as client:
                            sent=[]
                            for i in range(6):
                                ids=cases[i%len(cases)];marker,data=wire(ids,64);client.sendall(data);sent.append((ids,marker))
                            client.shutdown(socket.SHUT_WR)
                            for ids,marker in sent:keep(receive(client,ids,64,marker),"pipeline_halfclose")
                        bad_last=prompts[0].copy();bad_last[-1]=config.vocab_size
                        bad_first=prompts[0].copy();bad_first[0]=0xffffffff
                        with connect() as client:
                            sent=[]
                            for ids,bad in ((bad_last,True),(prompts[0],False),(bad_first,True),(prompts[0],False)):
                                marker,data=wire(ids,64);client.sendall(data);sent.append((ids,marker,bad))
                            for ids,marker,bad in sent:keep(receive(client,ids,64,marker,bad=bad),"pipeline_reject")
                        exchange(bad_first,64,"binary_reject",http=False,bad=True)
                        exchange(prompts[0],64,"after_reject",http=False)
                    else:
                        with connect() as client:
                            for i in range(8):
                                ids=prompts[i%len(prompts)]
                                keep(dict(request(client,ids,None,gen=1),input_ids=ids,gen=1,bad=False),"warmup")
                        cell_order=matrix[pair%4:]+matrix[:pair%4] if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison else matrix
                        for gen,concurrency in cell_order:
                            barrier=threading.Barrier(concurrency+1);ready=threading.Barrier(concurrency+1)
                            selected=[]
                            def lane(i):
                                with connect() as client:
                                    ready.wait(timeout=60);barrier.wait(timeout=60)
                                    for index in range(i,requests_per_cell,concurrency):
                                        ids=prompts[index%len(prompts)]
                                        row=dict(request(client,ids,None,gen=gen),input_ids=ids,gen=gen,bad=False,index=index,concurrency=concurrency)
                                        selected.append(row)
                            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                                futures=[pool.submit(lane,i) for i in range(concurrency)]
                                ready.wait(timeout=60)
                                before=worker_cpu_snapshot(p.pid) if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison else None
                                barrier.wait(timeout=60)
                                for future in futures:future.result()
                            cpu=worker_cpu_delta(before,worker_cpu_snapshot(p.pid)) if args.transport_comparison or args.copy_comparison or args.loop_comparison or args.loop_progress_comparison else {}
                            first=min(row["start_ns"] for row in selected);last=max(row["arrival_ns"][-1] for row in selected)
                            for row in selected:keep(row,"measured")
                            emit("cell",pair=pair,mode=mode,gen=gen,concurrency=concurrency,
                                requests=requests_per_cell,tokens=requests_per_cell*gen,first_ns=first,last_ns=last,output_tokens_per_s=requests_per_cell*gen/((last-first)/1e9),**cpu)
                    if args.correctness:
                        tids=[int(t.name) for t in Path(f"/proc/{p.pid}/task").iterdir() if int(t.name)!=p.pid]
                        assert tids
                        ret=ctypes.CDLL(None).tgkill(p.pid,tids[0],signal.SIGTERM);assert ret==0
                        emit("helper_signal",pair=pair,mode=mode,pid=p.pid,tid=tids[0])
                    else:p.terminate()
                    code=p.wait();log=logpath.read_text()
                    emit("executor",pair=pair,mode=mode,exit_code=code,stdout=log)
                    assert code==0 and "graph_error=0" in log and "launches=1 " in log,log[-6000:]
                    if args.dispatch_comparison:
                        calls={k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",log.split("resident vLLM stopped:",1)[1].splitlines()[0])}
                        assert calls["device_launch_calls"]==calls["device_graph_launches"]*(1 if mode.startswith("chained") else 2),calls
                        emit("dispatch_verified",pair=pair,mode=mode,device_launch_calls=calls["device_launch_calls"],model_steps=calls["device_graph_launches"])
                    if args.untraced:
                        batches=None
                        counters={k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",log.split("resident vLLM stopped:",1)[1].splitlines()[0])}
                        assert counters["admitted"]==counters["retired"]==len(rows)
                        assert counters["graph_error"]==counters["trace_overflows"]==0
                        checked=0
                        for row in rows:
                            if row["bad"]:assert row["tokens"]==[]
                            else:
                                assert len(row["tokens"])==row["gen"]
                                assert tuple(row["tokens"]) in reference_pool[(tuple(row["input_ids"]),row["gen"])],("unqualified complete output sequence",row)
                                checked+=len(row["tokens"])
                        emit("verified",pair=pair,mode=mode,requests=len(rows),tokens=checked,
                             rejections=sum(row["bad"] for row in rows),reference_scope="frozen complete-output-set membership; current batch trajectory unobserved",
                             reference_path=str(args.reference_path),native_counters=counters)
                    else:
                        refs=[json.loads(line) for line in Path(str(trace)+".references.jsonl").open()]
                        assert len(rows)==len(refs),(len(rows),len(refs))
                        by_marker={row["client_ns"]:row for row in refs};assert len(by_marker)==len(refs)
                        checked=0
                        for row in rows:
                            ref=by_marker.pop(row["client_ns"])
                            assert ref["prompt"]==row["input_ids"] and ref["gen"]==row["gen"] and bool(ref["bad"])==row["bad"],(row,ref)
                            assert ref["tokens"]==row["tokens"],(row,ref)
                            checked+=len(row["tokens"])
                        assert not by_marker
                        if frozen:
                            known=[row for row in rows if not row["bad"] and tuple(row["input_ids"]) in frozen]
                            emit("unpadded_fidelity_diagnostic",pair=pair,mode=mode,sequences=len(known),
                                equal_sequences=sum(row["tokens"]==frozen[tuple(row["input_ids"])][:row["gen"]] for row in known),
                                first_token_equal=sum(row["tokens"][0]==frozen[tuple(row["input_ids"])][0] for row in known))
                        events=[json.loads(line) for line in trace.open()]
                        assert events[-1]["overflows"]==0
                        if args.release_comparison:
                            releases=[e for e in events if e["event"]=="release"]
                            assert sum(e["batch"] for e in releases)==len(rows)
                            actual_max=max(e["batch"] for e in releases)
                            assert actual_max==1 if mode.startswith("serial") else actual_max>1
                            emit("release_verified",pair=pair,mode=mode,rounds=len(releases),max_prefix=actual_max)
                        batches=sorted({e["batch"] for e in events if e["event"]=="compute"})
                        queries=sorted({e["query"] for e in events if e["event"]=="compute"})
                        emit("verified",pair=pair,mode=mode,requests=len(rows),tokens=checked,
                            rejections=sum(row["bad"] for row in rows),actual_batches=batches,actual_queries=queries,reference_path=str(trace)+".references.jsonl")
                    if args.correctness:
                        network=nettrace.read_text();assert "resident vLLM ready" in network
                        active=network.split("resident vLLM ready",1)[1]
                        calls=r"\b(?:accept4?|recvfrom|recvmsg|recvmmsg|sendto|sendmsg|sendmmsg)\("
                        if kernel_mode:assert not re.search(calls,active),active
                        else:
                            for call in ("accept4","recvfrom","sendmsg"):assert re.search(r"\b"+call+r"\(",active),call
                        if mode.startswith("uring"):
                            serving=active.split("--- SIGTERM",1)[0]
                            if mode.startswith("uring_loop"):assert len(re.findall(r"\bio_uring_enter\(",serving))==1,serving
                            else:assert not re.search(r"\bio_uring_enter\(",serving),serving
                        emit("network_verified",pair=pair,mode=mode,trace_path=str(nettrace))
                    if mode.startswith("uring"):
                        counters=json.loads(next(line[7:] for line in log.splitlines() if line.startswith("RESULT ")))
                        assert counters["completed"]==counters["head"]==len(rows)
                        assert counters["error"]==counters["pending"]==counters["reserved"]==counters["dropped"]==counters["overflow"]==0
                        assert counters["accepted"]==counters["closed"]
                        if mode.startswith("uring_loop"):
                            assert counters["sqhead"]==counters["sqtail"] and counters["cqhead"]==counters["cqtail"]
                        else:assert counters["sqhead"]==counters["sqtail"]==counters["cqhead"]==counters["cqtail"]
                        assert counters["trace_count"]==len(rows) and counters["trace_overflow"]==max(0,len(rows)-128)
                        emit("uring_verified",pair=pair,mode=mode,counters=counters)
                    if kernel_mode:
                        r=subprocess.run(["./build/bpf_gpu_server_load","stats"],capture_output=True,text=True)
                        emit("stats",pair=pair,mode=mode,code=r.returncode,stdout=r.stdout,stderr=r.stderr);assert r.returncode==0
                        if args.untraced or mode.startswith("uring"):
                            assert f"llm_head={len(rows)}\n" in r.stdout and "llm_reserved=0x0" in r.stdout
                            slots=re.findall(r"llm\[(\d+)\]: state=(\d+)",r.stdout)
                            assert sorted(slots)==[(str(i),"0") for i in range(8)],slots
                    elif args.untraced:
                        native=re.search(r"userspace TCP stopped: accepted=(\d+) published=(\d+) completed=(\d+) rejected=(\d+)",log)
                        assert native and int(native[2])==int(native[3])==len(rows)
                        assert int(native[4])==sum(row["bad"] for row in rows)
                    assert not Path("/sys/module/bpf_gpu_server_tx").exists()
                    print(f"PASS pair={pair} mode={mode} requests={len(rows)} tokens={checked} batches={batches}",flush=True)
                    success=True
                finally:
                    if p is not None and p.poll() is None:p.terminate();emit("cleanup_executor",pair=pair,mode=mode,exit_code=p.wait())
                    if attached:
                        snapshot=subprocess.run(["./build/bpf_gpu_server_load","stats"],capture_output=True,text=True)
                        emit("cleanup_stats",pair=pair,mode=mode,code=snapshot.returncode,stdout=snapshot.stdout,stderr=snapshot.stderr)
                        r=subprocess.run(["./build/bpf_gpu_server_load","stream-detach"],capture_output=True,text=True)
                        emit("detach",pair=pair,mode=mode,code=r.returncode,stdout=r.stdout,stderr=r.stderr)
                        for name in ("bgs_ctl","bgs_stats","bgs_db"):Path("/sys/fs/bpf",name).unlink(missing_ok=True)
                    emit("executor_terminal",pair=pair,mode=mode,success=success)
        exit_code=0
    except BaseException as exc:
        emit("failure",kind=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:emit("terminal",exit_code=exit_code);raw.close()
    sys.exit(exit_code)
if __name__=="__main__":main()
