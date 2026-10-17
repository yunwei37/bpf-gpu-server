#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Passive loopback TCP metadata capture for the native Qwen token HTTP path."""
import argparse
import json
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import time
from resident_dispatch import command,connect,request,oracle_cases

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--executor",default="./build/qwen")
parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
args=parser.parse_args()
user_network=Path(args.executor).name=="qwen_graph_user_network"
weights="/workspaces/.cache/bpf-gpu-server/qwen25-05b-fp16.bin" if args.model=="Qwen/Qwen2.5-0.5B-Instruct" else args.model
cases,_=oracle_cases(8,name=args.model)
packets=[]; stop=threading.Event()
with socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(0x0800)) as capture:
    capture.bind(("lo",0)); capture.settimeout(.05)
    def collect():
        while not stop.is_set():
            try: packet,address=capture.recvfrom(65535)
            except socket.timeout: continue
            ns=time.monotonic_ns()
            if address[2]==socket.PACKET_OUTGOING or len(packet)<54 or packet[23]!=6: continue
            ip=14+(packet[14]&15)*4
            sport,dport,seq,ack=struct.unpack_from("!HHII",packet,ip)
            if 39403 not in (sport,dport): continue
            end=14+struct.unpack_from("!H",packet,16)[0]
            body=packet[ip+(packet[ip+12]>>4)*4:end]
            kind="request" if body.startswith(b"POST") else "header" if body.startswith(b"HTTP") else "tokens" if body else "empty"
            header_bytes=body.find(b"\r\n\r\n")+4 if kind=="header" and b"\r\n\r\n" in body else 0
            packets.append({"event":"packet","mono_ns":ns,"sport":sport,"dport":dport,
                "seq":seq,"ack":ack,"flags":packet[ip+13],"bytes":len(body),"kind":kind,"header_bytes":header_bytes,
                "token_bytes":len(body)-header_bytes if kind=="header" and header_bytes else len(body) if kind=="tokens" else 0})
    observer=threading.Thread(target=collect); observer.start()
    executor=None
    with tempfile.TemporaryDirectory(prefix="bpf-gpu-server-tcp-trace-") as tmp:
        logpath=Path(tmp)/"executor.log"
        try:
            if not user_network: command("./build/bpf_gpu_server_load","stream-attach")
            else: assert not Path("/sys/module/bpf_gpu_server_tx").exists()
            with logpath.open("w") as log:
                executor=subprocess.Popen([args.executor,weights,"60"],stdout=log,stderr=subprocess.STDOUT)
            while "resident Qwen ready" not in logpath.read_text():
                assert executor.poll() is None,logpath.read_text()
                time.sleep(.05)
            print(json.dumps({"event":"metadata","commit":command("git","rev-parse","HEAD"),"executor":args.executor,"model":args.model,"clock":"client and passive capture monotonic_ns; receive timestamps include observer scheduling"}),flush=True)
            with connect() as client:
                for case,(ids,expected) in enumerate(cases):
                    print(json.dumps({"event":"reply","case":case,**request(client,ids,expected)}),flush=True)
            time.sleep(.1)
        finally:
            stop.set(); observer.join()
            for packet in packets: print(json.dumps(packet),flush=True)
            if executor is not None:
                executor.terminate(); executor.wait(timeout=60)
                print(json.dumps({"event":"executor","log":logpath.read_text(),"stats":None if user_network else command("./build/bpf_gpu_server_load","stats")}),flush=True)
            command("./build/bpf_gpu_server_load","stream-detach")
            command("./build/bpf_gpu_server_load","detach","lo")
            for name in ("bgs_ctl","bgs_stats","bgs_db"):
                Path("/sys/fs/bpf",name).unlink(missing_ok=True)
