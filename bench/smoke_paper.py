#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-GPU smoke: normal builds, cuBLAS/cuFFT, both transports."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="New directory outside the source tree")
    args = ap.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        ap.error("--output must be outside the source tree")
    out.mkdir(parents=True, exist_ok=False)
    pins = Path("/sys/fs/bpf")
    if list(pins.glob("bgs_*")):
        raise RuntimeError("An existing transport owns BPF pins; preserve it and stop this smoke")
    build = out / "build"
    command = ["make", "BUILD=" + str(build), str(build / "server.bpf.o"),
               str(build / "bpf_gpu_server_load"), "graph-server-kernel-network", "graph-server-user-network-wait"]
    with (out / "build.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    env = dict(os.environ, BGS_BIND="127.0.0.1", PYTHONUNBUFFERED="1")
    results = []
    for operation in ["matmul", "fft"]:
        if operation == "matmul":
            inputs = [1.] * 64
            expected = [float(sum((8 * k + j) % 7 - 3 for k in range(8))) for i in range(8) for j in range(8)]
        else:
            inputs = [1., 0.] * 32
            expected = [32., 0.] + [0.] * 62
        for mode in ["native", "kernel"]:
            tag = operation + "_" + mode
            log_path = out / (tag + ".log")
            attached = False
            proc = None
            receipt = dict(operation=operation, transport=mode, requests=0)
            try:
                if mode == "kernel":
                    attached = True
                    result = subprocess.run([str(build / "bpf_gpu_server_load"), "stream-uring-attach"], cwd=ROOT, env=env,
                                            capture_output=True, text=True)
                    (out / (tag + ".attach.txt")).write_text(result.stdout + result.stderr)
                    result.check_returncode()
                library = build / ("graph_server_kernel_network.so" if mode == "kernel" else "graph_server_user_network_wait.so")
                cmd = [sys.executable, str(ROOT / "examples/torch_graph_server.py"), str(library),
                       operation, "--seconds", "45"]
                with log_path.open("w") as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 120
                    while "graph backend ready:" not in log_path.read_text():
                        if proc.poll() is not None:
                            raise RuntimeError(tag + " stopped before ready; see its log")
                        if time.monotonic() >= deadline:
                            raise RuntimeError(tag + " did not become ready")
                        time.sleep(.1)
                    # Exact independent constant-input oracles; include persistent reuse.
                    for connection in range(4):
                        with socket.create_connection(("127.0.0.1", 39403), timeout=15) as sock:
                            for request in range(4):
                                frame = struct.pack("<IIIIQ", 0x514c4d51, 64, 64, 0, time.monotonic_ns())
                                sock.sendall(frame + struct.pack("<64f", *inputs))
                                reply = bytearray()
                                while len(reply) < 256:
                                    chunk = sock.recv(256 - len(reply))
                                    if not chunk:
                                        raise RuntimeError("short reply")
                                    reply.extend(chunk)
                                assert list(struct.unpack("<64f", reply)) == expected
                                receipt["requests"] += 1
                    proc.send_signal(signal.SIGINT)
                    receipt["exit"] = proc.wait(timeout=60)
                    assert receipt["exit"] == 0
                text = log_path.read_text()
                assert "graph_error=0" in text
                receipt["status"] = "PASS"
            except BaseException as error:
                receipt.update(status="FAIL", error=repr(error))
                raise
            finally:
                if proc is not None and proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=60)
                if attached:
                    det = subprocess.run([str(build / "bpf_gpu_server_load"), "stream-detach"], cwd=ROOT, env=env,
                                         capture_output=True, text=True)
                    (out / (tag + ".detach.txt")).write_text(det.stdout + det.stderr)
                    for name in ["bgs_ctl", "bgs_stats", "bgs_db"]:
                        (pins / name).unlink(missing_ok=True)
                    assert det.returncode == 0 and not list(pins.glob("bgs_*"))
                results.append(receipt)
                (out / "smoke.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(dict(cells=len(results), checked_requests=sum(x["requests"] for x in results), status="PASS")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
