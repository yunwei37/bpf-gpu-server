#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Matched HF graph dispatch/network controls on the token HTTP path."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
from resident_qwen import oracle_cases


def command(*args):
    return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()


def cpu_seconds(pid):
    fields=Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()
    return (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")


_libc=ctypes.CDLL(None)
_libc.clock_getcpuclockid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
_libc.clock_getcpuclockid.restype=ctypes.c_int


def worker_cpu_snapshot(pid):
    # POSIX process CPU clock includes all executor threads. TX uses the
    # kernel scheduler's adjusted task accounting, exposed read-only.
    # Neither counter attributes RX/softirq, BPF execution or all TCP work.
    begin=time.monotonic_ns()
    clock=ctypes.c_int()
    error=_libc.clock_getcpuclockid(pid,ctypes.byref(clock))
    if error: raise OSError(error,os.strerror(error))
    process=time.clock_gettime_ns(clock.value)
    tx_path=Path("/sys/module/bpf_gpu_server_tx/parameters/tx_cpu_ns")
    tx=int(tx_path.read_text()) if Path("/sys/module/bpf_gpu_server_tx").exists() else 0
    return {"sample_begin_ns":begin,"sample_end_ns":time.monotonic_ns(),
            "executor_cpu_ns":process,"kernel_tx_cpu_ns":tx}


def worker_cpu_delta(before,after):
    process=after["executor_cpu_ns"]-before["executor_cpu_ns"]
    tx=after["kernel_tx_cpu_ns"]-before["kernel_tx_cpu_ns"]
    assert process>=0 and tx>=0,(before,after)
    return {"worker_cpu_before":before,"worker_cpu_after":after,
            "executor_clock_cpu_ns":process,"kernel_tx_cpu_ns":tx,
            "known_worker_cpu_ns":process+tx}


def host_cpu_ticks():
    # Global host counters include clients, kernel work and other workloads.
    return [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]


def gpu_state():
    return command("nvidia-smi","--query-gpu=timestamp,name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu","--format=csv,noheader,nounits")


def request(client,ids,expected,gen=None):
    # A dynamic-batch backend validates against independent trajectory replay.
    if expected is not None: gen=len(expected)
    assert isinstance(gen,int) and gen>0
    client_ns=time.monotonic_ns()
    body=struct.pack("<IIIIQ",0x514c4d51,len(ids),gen,0,client_ns)+struct.pack(f"<{len(ids)}I",*ids)
    wire=f"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body
    start=time.monotonic_ns()
    client.sendall(wire)
    header=b""; data=b""; arrivals=[]
    while len(data)<4*gen:
        chunk=client.recv(4096)
        arrived=time.monotonic_ns()
        assert chunk,"short reply"
        if header is not None:
            header+=chunk
            if b"\r\n\r\n" not in header: continue
            head,chunk=header.split(b"\r\n\r\n",1)
            assert head.startswith(b"HTTP/1.1 200 OK\r\n"),head
            assert f"Content-Length: {4*gen}".encode() in head,head
            header=None
        data+=chunk
        arrivals.extend([arrived]*((len(data)//4)-len(arrivals)))
    assert len(data)==4*gen,data
    actual=list(struct.unpack(f"<{gen}I",data))
    if expected is not None: assert actual==expected,(ids,actual,expected)
    return {"client_ns":client_ns,"start_ns":start,"arrival_ns":arrivals,"tokens":actual,
            "ttft_ms":(arrivals[0]-start)/1e6,
            "latency_ms":(arrivals[-1]-start)/1e6,
            "tpot_ms":(arrivals[-1]-arrivals[0])/(gen-1)/1e6 if gen>1 else None}


def connect():
    client=socket.create_connection((os.environ.get("BGS_BIND","127.0.0.1"),39403),timeout=60)
    client.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    return client


def cell(pid,cases,gen,concurrency,count,rows):
    # Connections and thread creation precede both the timer and CPU window.
    barrier=threading.Barrier(concurrency+1)
    ready=threading.Barrier(concurrency+1)
    def worker(worker_id):
        with connect() as client:
            ready.wait(timeout=60)
            barrier.wait(timeout=60)
            for index in range(worker_id,count,concurrency):
                case=index%len(cases); ids,expected=cases[case]
                row=request(client,ids,expected[:gen]); row.update(index=index,case=case)
                rows.append(row)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(worker,i) for i in range(concurrency)]
        ready.wait(timeout=60)
        before=cpu_seconds(pid); host_before=host_cpu_ticks(); worker_before=worker_cpu_snapshot(pid); start=time.monotonic_ns()
        barrier.wait(timeout=60)
        for future in futures: future.result()
        finish=time.monotonic_ns(); worker_after=worker_cpu_snapshot(pid); host_after=host_cpu_ticks(); after=cpu_seconds(pid)
    first=min(row["start_ns"] for row in rows)
    last=max(row["arrival_ns"][-1] for row in rows)
    return {
        "window_ns":finish-start,"executor_cpu_s":after-before,
        **worker_cpu_delta(worker_before,worker_after),
        "host_cpu_ticks_before":host_before,"host_cpu_ticks_after":host_after,
        "host_cpu_work_s":sum(host_after[i]-host_before[i] for i in (0,1,2,5,6))/os.sysconf("SC_CLK_TCK"),
        "output_tokens_per_s":count*gen/((last-first)/1e9)}


def analyze(path):
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    meta=next(row for row in records if row["event"]=="metadata")
    cells=[row for row in records if row["event"]=="cell"]
    samples=[row for row in records if row["event"]=="request"]
    expected_cells=meta["repetitions"]*len(meta["executors"])*len(meta["matrix"])
    assert len(cells)==expected_cells,(len(cells),expected_cells)
    assert len(samples)==expected_cells*meta["requests_per_cell"]
    assert len([row for row in records if row["event"]=="executor"])==len(meta["executors"])*meta["repetitions"]
    for row in samples:
        assert row["tokens"]==meta["cases"][row["case"]][1][:row["gen"]],row
        assert row["ttft_ms"]==(row["arrival_ns"][0]-row["start_ns"])/1e6,row
    for block in cells:
        selected=[row for row in samples if all(row[key]==block[key] for key in ("pair","mode","gen","concurrency"))]
        assert sorted(row["index"] for row in selected)==list(range(meta["requests_per_cell"])),block
    def percentile(values,q):
        ordered=sorted(values); pos=(len(ordered)-1)*q
        low=math.floor(pos); high=math.ceil(pos)
        return ordered[low]+(ordered[high]-ordered[low])*(pos-low)
    modes=tuple(meta["executors"])
    for gen,concurrency in meta["matrix"]:
        pair_means={}
        for mode in modes:
            selected=[row for row in samples if row["mode"]==mode and row["gen"]==gen and row["concurrency"]==concurrency]
            selected_cells=[row for row in cells if row["mode"]==mode and row["gen"]==gen and row["concurrency"]==concurrency]
            ttft=[row["ttft_ms"] for row in selected]
            tpot=[row["tpot_ms"] for row in selected if row["tpot_ms"] is not None]
            pair_means[mode]=[statistics.mean(row["ttft_ms"] for row in selected if row["pair"]==pair) for pair in range(meta["repetitions"])]
            print(json.dumps({"mode":mode,"gen":gen,"concurrency":concurrency,"requests":len(selected),
                "ttft_mean_ms":statistics.mean(ttft),"ttft_p50_ms":percentile(ttft,.5),"ttft_p99_ms":percentile(ttft,.99),
                "tpot_mean_ms":statistics.mean(tpot) if tpot else None,
                "completion_mean_ms":statistics.mean(row["latency_ms"] for row in selected),
                "host_cpu_work_s_per_cell":[row.get("host_cpu_work_s") for row in selected_cells],
                "executor_cpu_s_per_cell":[row["executor_cpu_s"] for row in selected_cells],
                "known_worker_cpu_ns_per_cell":[row.get("known_worker_cpu_ns") for row in selected_cells],
                "kernel_tx_cpu_ns_per_cell":[row.get("kernel_tx_cpu_ns") for row in selected_cells],
                "tokens_per_s":[row["output_tokens_per_s"] for row in selected_cells]}))
        for control in modes[1:]:
            differences=[host-resident for host,resident in zip(pair_means[control],pair_means["resident"])]
            mean=statistics.mean(differences)
            # Five independent paired process blocks; Student t, df=4, two-sided 95%.
            margin=2.776445105*statistics.stdev(differences)/math.sqrt(5) if len(differences)==5 else None
            print(json.dumps({"control":control,"gen":gen,"concurrency":concurrency,("paired_host_minus_resident_ttft_ms" if control=="host-launch" else "paired_user_network_minus_resident_ttft_ms"):differences,
                "difference_mean_ms":mean,"difference_t95_ms":[mean-margin,mean+margin] if margin is not None else None}))
            differences=[next(row["output_tokens_per_s"] for row in cells if row["pair"]==pair and row["mode"]=="resident" and row["gen"]==gen and row["concurrency"]==concurrency)-
                         next(row["output_tokens_per_s"] for row in cells if row["pair"]==pair and row["mode"]==control and row["gen"]==gen and row["concurrency"]==concurrency) for pair in range(meta["repetitions"])]
            mean=statistics.mean(differences)
            margin=2.776445105*statistics.stdev(differences)/math.sqrt(5) if len(differences)==5 else None
            print(json.dumps({"control":control,"gen":gen,"concurrency":concurrency,("paired_resident_minus_host_tps" if control=="host-launch" else "paired_resident_minus_user_network_tps"):differences,
                "difference_mean_tps":mean,"difference_t95_tps":[mean-margin,mean+margin] if margin is not None else None}))
            if all("known_worker_cpu_ns" in row for row in cells):
                differences=[next(row["known_worker_cpu_ns"] for row in cells if row["pair"]==pair and row["mode"]==control and row["gen"]==gen and row["concurrency"]==concurrency)-
                             next(row["known_worker_cpu_ns"] for row in cells if row["pair"]==pair and row["mode"]=="resident" and row["gen"]==gen and row["concurrency"]==concurrency) for pair in range(meta["repetitions"])]
                mean=statistics.mean(differences)
                margin=2.776445105*statistics.stdev(differences)/math.sqrt(5) if len(differences)==5 else None
                divisor=meta["requests_per_cell"]*1000
                print(json.dumps({"control":control,"gen":gen,"concurrency":concurrency,
                    "paired_control_minus_resident_worker_cpu_us_per_request":[value/divisor for value in differences],
                    "difference_mean_worker_cpu_us_per_request":mean/divisor,
                    "difference_t95_worker_cpu_us_per_request":[(mean-margin)/divisor,(mean+margin)/divisor] if margin is not None else None}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--gen",type=int,default=8,choices=range(2,65),help="long output length; paired with one-token requests")
    parser.add_argument("--analyze",action="store_true")
    parser.add_argument("--resident-executor",default="./build/qwen")
    parser.add_argument("--host-executor",default="./build/qwen_host_launch")
    parser.add_argument("--user-network-executor",help="same resident GPU compute with userspace network control")
    parser.add_argument("--user-network-wait-executor",help="additional native epoll_pwait2 network control")
    parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct",help="HF checkpoint for graph executors")
    args=parser.parse_args()
    if args.analyze: analyze(args.output); return
    control="user-network" if args.user_network_executor else "host-launch"
    executors={"resident":args.resident_executor,control:args.user_network_executor or args.host_executor}
    if args.model!="Qwen/Qwen2.5-0.5B-Instruct" and any("qwen_graph" not in Path(binary).name for binary in executors.values()):
        parser.error("--model requires graph executors")
    if args.user_network_wait_executor:
        if not args.user_network_executor: parser.error("--user-network-wait-executor requires --user-network-executor")
        executors["user-network-wait"]=args.user_network_wait_executor
    weights="/workspaces/.cache/bpf-gpu-server/qwen25-05b-fp16.bin" if args.model=="Qwen/Qwen2.5-0.5B-Instruct" else args.model
    cases,_=oracle_cases(args.gen,name=args.model)
    matrix=[(1,1)] if args.preflight else [(1,1),(args.gen,1),(1,8),(args.gen,8)]
    repetitions=1 if args.preflight else 5
    count=4 if args.preflight else 64
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as raw:
        def emit(event,**values):
            raw.write(json.dumps({"event":event,**values})+"\n"); raw.flush()
        import torch, transformers
        emit("metadata",utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),commit=command("git","rev-parse","HEAD"),
             executors=executors,model=args.model,server=[os.environ.get("BGS_BIND","127.0.0.1"),39403],
             network_control={name:("native epoll_pwait2 60000ns wait" if name=="user-network-wait" else "native epoll busy polling") for name in executors if name.startswith("user-network")},
             worker_cpu="POSIX executor process CPU + adjusted kernel TX task CPU; excludes RX/softirq/BPF and other TCP work; read brackets retained",
             kernel=platform.release(),torch=torch.__version__,transformers=transformers.__version__,
             nvcc=command("/usr/local/cuda/bin/nvcc","--version"),gpu=gpu_state(),cpu_tick_hz=os.sysconf("SC_CLK_TCK"),
             cases=cases,matrix=matrix,repetitions=repetitions,requests_per_cell=count,
             timing="client monotonic; tokens completed in one recv share arrival timestamp; process CPU excludes kernel TX/softirq",
             host_cpu="global /proc/stat user+nice+system+irq+softirq; includes clients and unrelated host workloads; approximate cell window, not service attribution")
        for pair in range(repetitions):
            order=tuple(executors)
            if len(order)==2:
                modes=order if pair%2==0 else tuple(reversed(order))
            else:
                modes=order[pair%len(order):]+order[:pair%len(order)]
                if pair%2: modes=tuple(reversed(modes))
            cells=matrix[pair%len(matrix):]+matrix[:pair%len(matrix)]
            for mode in modes:
                executor=None
                with tempfile.TemporaryDirectory(prefix="bpf-gpu-server-dispatch-") as tmp:
                    logpath=Path(tmp)/"executor.log"
                    try:
                        if not mode.startswith("user-network"):command("./build/bpf_gpu_server_load","stream-attach")
                        else:assert not Path("/sys/module/bpf_gpu_server_tx").exists(),"kernel TX must be absent for network control"
                        binary=executors[mode]
                        started=time.monotonic()
                        with logpath.open("w") as log:
                            executor=subprocess.Popen([binary,weights,"300"],stdout=log,stderr=subprocess.STDOUT)
                        while "resident Qwen ready" not in logpath.read_text():
                            assert executor.poll() is None,logpath.read_text()
                            time.sleep(.05)
                        emit("bootstrap",pair=pair,mode=mode,seconds=time.monotonic()-started)
                        with connect() as client:
                            for gen in (1,args.gen):
                                for ids,expected in cases: request(client,ids,expected[:gen])
                        before=cpu_seconds(executor.pid); worker_before=worker_cpu_snapshot(executor.pid); idle_start=time.monotonic_ns(); time.sleep(1)
                        worker_after=worker_cpu_snapshot(executor.pid)
                        emit("idle",pair=pair,mode=mode,window_ns=time.monotonic_ns()-idle_start,executor_cpu_s=cpu_seconds(executor.pid)-before,
                             **worker_cpu_delta(worker_before,worker_after))
                        for gen,concurrency in cells:
                            gpu_before=gpu_state()
                            rows=[]
                            try:
                                summary=cell(executor.pid,cases,gen,concurrency,count,rows)
                            finally:
                                for row in sorted(rows,key=lambda row:row["index"]):
                                    emit("request",pair=pair,mode=mode,gen=gen,concurrency=concurrency,**row)
                            gpu_after=gpu_state()
                            emit("cell",pair=pair,mode=mode,gen=gen,concurrency=concurrency,requests=count,
                                 gpu_before=gpu_before,gpu_after=gpu_after,**summary)
                            print(f"pair={pair} mode={mode} gen={gen} c={concurrency} requests={count} exact_tokens={count*gen} cpu_s={summary['executor_cpu_s']:.3f}",flush=True)
                        executor.terminate(); assert executor.wait(timeout=60)==0,logpath.read_text()
                        log=logpath.read_text(); launches=1 if mode=="resident" or mode.startswith("user-network") else 8+count*len(matrix)
                        dispatch="resident" if mode.startswith("user-network") else mode
                        assert f"dispatch={dispatch} launches={launches}" in log,log
                        if mode.startswith("user-network"):
                            requests=8+count*len(matrix)
                            assert f"published={requests} completed={requests} rejected=0" in log,log
                            assert "network=userspace epoll" in log,log
                            expected_wait="epoll_pwait2 timeout_ns=60000" if mode=="user-network-wait" else "busy epoll_wait timeout_ms=0"
                            assert "userspace completion wait="+expected_wait in log,log
                            assert not Path("/sys/module/bpf_gpu_server_tx").exists()
                        if "device_graph_launches=" in log:
                            model_steps=4*(1+args.gen)+count*sum(gen for gen,_ in matrix)
                            assert f"device_graph_launches={model_steps} graph_error=0 banks=8" in log,log
                            if not args.preflight: assert "max_active=8" in log,log
                        emit("executor",pair=pair,mode=mode,log=log,stats=command("./build/bpf_gpu_server_load","stats") if not mode.startswith("user-network") else None)
                    except Exception as error:
                        emit("failure",pair=pair,mode=mode,error=repr(error),
                             log=logpath.read_text() if logpath.exists() else "",
                             stats=command("./build/bpf_gpu_server_load","stats") if not mode.startswith("user-network") else None)
                        raise
                    finally:
                        if executor is not None and executor.poll() is None:
                            executor.terminate(); executor.wait(timeout=60)
                        command("./build/bpf_gpu_server_load","stream-detach")
                        command("./build/bpf_gpu_server_load","detach","lo")
                        for name in ("bgs_ctl","bgs_stats","bgs_db"):
                            Path("/sys/fs/bpf",name).unlink(missing_ok=True)
    analyze(output)


if __name__=="__main__": main()
