#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: CPU instructions per generated token on the LLM path.
#
# The resident executor's instruction stream is dominated by its poll loop
# between requests, so an idle-subtracted number is meaningless. Instead drive
# it into saturation (always a request queued) for a fixed window, count its
# retired instructions over that window and divide by the tokens produced.
# With no idle gaps, the count is the real per-token CPU cost (Python HF
# decode + page writes + the tiny poll between back-to-back requests).
#
#   python3 tools/insn_token.py --pid <executor-pid> --threads 16 --secs 6
import argparse, os, socket, struct, threading, time
from transformers import AutoTokenizer

from llm_executor import BGS_LLM_SLOTS, LLM_HEAD_OFF, open_page, Page

BGS_LLM_PORT = 39402
BGS_LLM_MAGIC = 0x514c4d51
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERFCOUNT = os.path.join(ROOT, "build", "perfcount")


def count(pid, ms):
    out = os.popen(f"{PERFCOUNT} {pid} {ms}").read()
    return int(out.split()[0].split("=")[1])


def one_request(page, ids, n_gen, dest, counter):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_prompt = len(ids)
    client_ns = time.monotonic_ns()
    hdr = struct.pack("<IIIIQ", BGS_LLM_MAGIC, n_prompt, n_gen, 0, client_ns)
    s.sendto(hdr + struct.pack(f"<{n_prompt}I", *ids), dest)
    deadline = time.perf_counter() + 30
    slot = None
    while time.perf_counter() < deadline and slot is None:
        head = page.u32(LLM_HEAD_OFF)
        for k in range(BGS_LLM_SLOTS):
            idx = (head - 1 - k) % BGS_LLM_SLOTS
            if page.slot(idx)[7] == client_ns:
                slot = idx
                break
    if slot is None:
        return
    last = 0
    while time.perf_counter() < deadline:
        p = page.slot(slot)[3]
        if p > last:
            last = p
            if p >= n_gen:
                break
    counter[0] += last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--secs", type=int, default=6)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(a.prompt).input_ids
    mm, base = open_page()
    page = Page(mm, base)
    dest = ("127.0.0.1", BGS_LLM_PORT)

    counter = [0]
    stop = [False]

    def worker():
        while not stop[0]:
            one_request(page, ids, a.gen, dest, counter)

    ths = [threading.Thread(target=worker) for _ in range(a.threads)]
    for t in ths:
        t.start()
    time.sleep(0.3)                      # past first-touch / model warm
    counter[0] = 0                       # count tokens only in the window
    t0 = time.perf_counter()
    ins = count(a.pid, a.secs * 1000)
    wall = time.perf_counter() - t0
    stop[0] = True
    for t in ths:
        t.join()
    tok_n = counter[0]
    print(f"threads={a.threads} gen={a.gen} wall={wall:.2f}s "
          f"tokens={tok_n} ({tok_n/wall:.0f} tok/s)")
    print(f"executor instructions={ins/1e9:.3f} G "
          f"instructions/token={ins/tok_n/1e6:.1f}e6 M")


if __name__ == "__main__":
    main()
