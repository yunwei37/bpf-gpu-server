#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local pinned-model LLM smoke: resident16/kernel16/resident32/official."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Downloaded pinned Qwen2.5-0.5B or SmolLM2-135M checkpoint")
    ap.add_argument("--output", type=Path, required=True, help="New directory outside source tree")
    args = ap.parse_args()
    model, out = args.model.resolve(), args.output.resolve()
    if not (model / "config.json").is_file() or out.is_relative_to(ROOT):
        ap.error("Need a downloaded model and a new output directory outside source")
    out.mkdir(parents=True, exist_ok=False)
    pins = Path("/sys/fs/bpf")
    assert not list(pins.glob("bgs_*")), "Preserve the existing BPF owner"
    archive = ROOT / "bench/results/official_sustained_20261016"
    model_config = json.loads((model / "config.json").read_text())
    candidates = [json.loads(p.read_text()) for p in
                  (ROOT / "bench/results/kernel_network_integration_20261017").glob("*_frozen.json")]
    candidates = [c for c in candidates
                  if c["frozen_model"]["vocab"] == model_config["vocab_size"]]
    if len(candidates) != 1:
        ap.error("Need one of the two pinned snapshots; frozen prompt/output references must match the checkpoint")
    cfg = candidates[0]
    config = out / "config.json"
    config.write_text(json.dumps(cfg))
    key = out / "private-test-key"
    key.write_text(secrets.token_urlsafe(24))
    key.chmod(0o600)
    env = dict(os.environ, BGS_BIND="127.0.0.1", PYTHONPATH=str(ROOT / "executor"),
               PYTHONUNBUFFERED="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               TORCH_EXTENSIONS_DIR=str(out / "extensions"), VLLM_CACHE_ROOT=str(out / "vllm-cache"),
               VLLM_API_KEY=key.read_text())
    receipts = []
    try:
        for slots in [16, 32]:
            build = out / ("build%d" % slots)
            cmd = ["make", "BUILD=" + str(build), "CPPFLAGS=-DBGS_LLM_SLOTS=%d" % slots,
                   "VLLM_PYTHON=" + sys.executable, str(build / "server.bpf.o"), str(build / "bpf_gpu_server_load"),
                   "vllm-mixed-user-network-wait"]
            if slots == 16:
                cmd.append("vllm-mixed-kernel-network")
            with (out / ("build%d.log" % slots)).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for tag, mode, slots in [("native16", "native", 16), ("kernel16", "kernel", 16),
                                 ("native32", "native", 32), ("official", "official", 0)]:
            build = out / ("build%d" % (slots or 16))
            attached = False
            server = None
            receipt = dict(condition=tag, mode=mode)
            try:
                if mode == "kernel":
                    attached = True
                    z = subprocess.run([str(build / "bpf_gpu_server_load"), "stream-uring-attach"], cwd=ROOT, env=env,
                                       capture_output=True, text=True)
                    (out / (tag + ".attach.txt")).write_text(z.stdout + z.stderr)
                    z.check_returncode()
                if mode == "official":
                    cmd = [str(Path(sys.executable).parent / "vllm"), "serve", str(model), "--tokens-only",
                           "--dtype", "half", "--host", "127.0.0.1", "--port", "39404"]
                else:
                    name = "vllm_mixed_kernel_network" if mode == "kernel" else "vllm_mixed_user_network_wait"
                    cmd = [str(build / name), str(model), "600"]
                path = out / (tag + ".server.log")
                with path.open("w") as log:
                    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 600
                    needle = "Application startup complete" if mode == "official" else "resident vLLM ready:"
                    while needle not in path.read_text():
                        if server.poll() is not None:
                            raise RuntimeError(tag + " stopped before ready")
                        if time.monotonic() >= deadline:
                            raise RuntimeError(tag + " did not become ready")
                        time.sleep(.25)
                    for warm in [True, False]:
                        output = out / (tag + (".warmup.jsonl" if warm else ".client.jsonl"))
                        client = [sys.executable, str(archive / "official_remote_client.py"), "--host", "127.0.0.1",
                                  "--config", str(config), "--offsets", str(archive / "remote_offsets.json"),
                                  "--output", str(output), "--pair", "0", "--rate", "192", "--mode", mode,
                                  "--key-file", str(key), "--preflight"]
                        if warm:
                            client.append("--warmup")
                        with (out / (output.stem + ".driver.log")).open("w") as driver:
                            subprocess.run(client, cwd=ROOT, env=env, stdout=driver, stderr=subprocess.STDOUT, check=True)
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    requests = [x for x in rows if x["event"] == "request"]
                    assert rows[-1]["event"] == "client_terminal" and rows[-1]["exit"] == 0
                    assert len(requests) == 32 and all(x["outcome"] == "success" and len(x["tokens"]) == 64 for x in requests)
                    receipt.update(requests=len(requests), checked_tokens=sum(len(x["tokens"]) for x in requests),
                                   finite_reference_disagreements=sum(not x["reference_set_match"] for x in requests),
                                   status="PASS")
                    server.send_signal(signal.SIGINT)
                    receipt["server_exit"] = server.wait(timeout=90)
                    assert receipt["server_exit"] == 0
                    if mode != "official":
                        assert "graph_error=0" in path.read_text()
            except BaseException as error:
                receipt.update(status="FAIL", error=repr(error))
                raise
            finally:
                if server is not None and server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    server.wait(timeout=90)
                if attached:
                    z = subprocess.run([str(build / "bpf_gpu_server_load"), "stream-detach"], cwd=ROOT, env=env,
                                       capture_output=True, text=True)
                    (out / (tag + ".detach.txt")).write_text(z.stdout + z.stderr)
                    for name in ["bgs_ctl", "bgs_stats", "bgs_db"]:
                        (pins / name).unlink(missing_ok=True)
                    assert z.returncode == 0 and not list(pins.glob("bgs_*"))
                receipts.append(receipt)
                (out / "smoke.json").write_text(json.dumps(receipts, indent=2))
    finally:
        key.unlink(missing_ok=True)
    print(json.dumps(dict(cells=len(receipts), status="PASS")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
