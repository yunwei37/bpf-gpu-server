#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Matched HF Qwen correctness control: userspace send vs kernel TCP TX.
This is NOT a resident-GPU or HTTP benchmark. Loopback, pretokenized requests,
fp16 eager HF greedy batch-1 inference in both modes. Run from repository root.
"""
import ctypes
import gc
import os
import socket
import struct
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ["The capital of France is", "What is 2 plus 2?",
           "Write a short greeting.", "Linux is"]


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt).input_ids
        with torch.no_grad():
            out = model(torch.tensor([ids], device="cuda"), use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[:, -1].argmax(-1))
            expected = [nxt]
            for _ in range(63):
                out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[:, -1].argmax(-1))
                expected.append(nxt)
        for n in (1, 16, 64):
            cases.append((ids, expected[:n]))
    print(f"oracle: {len(cases)} prompt/length cases, exact greedy IDs; torch={torch.__version__}", flush=True)
    del model, out, past
    gc.collect()
    torch.cuda.empty_cache()
    env = dict(os.environ, PYTHONPATH="executor")
    for mode in ("userspace", "kernel"):
        subprocess.run(["./build/bpf_gpu_server_load", "detach", "lo"], check=True)
        subprocess.run(["./build/bpf_gpu_server_load", "attach", "lo"], check=True)
        loaded = False
        executor = None
        log_path = f"/tmp/bpf-gpu-server-qwen-{mode}.log"
        try:
            if mode == "kernel":
                subprocess.run(["./build/bpf_gpu_server_load", "tx-load"], check=True)
                loaded = True
            args = ["python3", "executor/llm_executor.py", "--tcp", "--seconds", "300"]
            if mode == "kernel":
                args.append("--no-send")
            with open(log_path, "w") as log:
                executor = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while "TCP listener" not in open(log_path).read():
                if executor.poll() is not None or time.monotonic() - start > 300:
                    raise AssertionError(open(log_path).read())
                time.sleep(0.2)
            samples = []
            for repetition in range(2):
                for trial, (ids, expected) in enumerate(cases):
                    with socket.create_connection(("127.0.0.1", 39403), timeout=30) as client:
                        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        marker = time.monotonic_ns()
                        wire = struct.pack("<IIIIQ", 0x514c4d51, len(ids), len(expected), 0, marker)
                        wire += struct.pack(f"<{len(ids)}I", *ids)
                        start = time.perf_counter()
                        client.sendall(wire)
                        data, stamps = b"", []
                        while len(data) < 4 * len(expected):
                            chunk = client.recv(4 - len(data) % 4)
                            assert chunk, "short reply"
                            data += chunk
                            if len(data) % 4 == 0:
                                stamps.append(time.perf_counter())
                        actual = list(struct.unpack(f"<{len(expected)}I", data))
                        assert actual == expected, (mode, repetition, trial, actual, expected)
                        ttft = (stamps[0] - start) * 1e3
                        tpot = (stamps[-1] - stamps[0]) / max(1, len(stamps) - 1) * 1e3
                        samples.append((ttft, tpot))
                        print(f"{mode} rep={repetition} case={trial} gen={len(expected)} exact=True TTFT_ms={ttft:.3f} TPOT_ms={tpot:.3f}", flush=True)
            print(f"PASS {mode}: {len(samples)} exact replies; no requests failed", flush=True)
        finally:
            if executor is not None:
                executor.terminate()
                executor.wait()
                print(open(log_path).read(), flush=True)
            if loaded:
                lib = ctypes.CDLL(None, use_errno=True)
                ret = lib.syscall(176, b"bpf_gpu_server_tx", 0)  # x86_64 delete_module
                assert ret == 0, os.strerror(ctypes.get_errno())
            subprocess.run(["./build/bpf_gpu_server_load", "detach", "lo"], check=True)


if __name__ == "__main__":
    main()
