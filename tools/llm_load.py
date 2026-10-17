#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: LLM-path concurrent-load bench.
#
# Drives K concurrent requests at the resident executor and reports, per
# concurrency level, the aggregate token goodput and per-request TTFT/TPOT.
# The executor currently serves one slot at a time, so this measures how much
# queueing (GPU host-induced bubble) appears as concurrency rises — the SLO
# picture of the current resident design.
import argparse, statistics, struct, socket, threading, time
from transformers import AutoTokenizer

from llm_executor import BGS_LLM_SLOTS, LLM_HEAD_OFF, open_page, Page, BGS_PENDING

BGS_LLM_PORT = 39402
BGS_LLM_MAGIC = 0x514c4d51


def one_request(page, dest, ids, n_gen, out):
    n_prompt = len(ids)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_ns = time.monotonic_ns()
    hdr = struct.pack("<IIIIQ", BGS_LLM_MAGIC, n_prompt, n_gen, 0, client_ns)
    payload = hdr + struct.pack(f"<{n_prompt}I", *ids)
    t_send = time.perf_counter()
    s.sendto(payload, dest)

    slot = None
    t_search_end = t_send + 5.0
    while time.perf_counter() < t_search_end and slot is None:
        head = page.u32(LLM_HEAD_OFF)
        for k in range(BGS_LLM_SLOTS):
            idx = (head - 1 - k) % BGS_LLM_SLOTS
            if page.slot(idx)[7] == client_ns:
                slot = idx
                break
    deadline = t_send + 60
    if slot is None:
        out.append(None)
        return
    stamps = []
    last = 0
    while time.perf_counter() < deadline:
        v = page.slot(slot)
        produced = v[3]
        if produced > last:
            now = time.perf_counter()
            while last < produced:
                stamps.append(now)
                last += 1
            if produced >= n_gen:
                break
    if not stamps:
        out.append(None)
        return
    out.append(((stamps[0] - t_send) * 1e6,
                (stamps[-1] - stamps[0]) / max(1, len(stamps) - 1) * 1e6,
                len(stamps)))


def run_level(page, dest, ids, n_gen, k, rounds):
    # warm the page path once so the first level is not skewed by first-touch
    _run_level(page, dest, ids, n_gen, 1, 1)
    return _run_level(page, dest, ids, n_gen, k, rounds)


def _run_level(page, dest, ids, n_gen, k, rounds):
    results = []
    lock = threading.Lock()

    def worker():
        for _ in range(rounds):
            out = []
            one_request(page, dest, ids, n_gen, out)
            with lock:
                results.append(out[0])

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(k)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r]
    return ok, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--levels", default="1 2 4 8")
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(a.prompt).input_ids
    mm, base = open_page()
    page = Page(mm, base)
    dest = ("127.0.0.1", BGS_LLM_PORT)

    print(f"prompt_tok={len(ids)} gen={a.gen} rounds/thread={a.rounds}")
    print(f"{'conc':>4} {'req':>4} {'tok/s':>8} {'TTFT p50':>9} "
          f"{'TTFT p99':>9} {'TPOT p50':>9}")
    for k in (int(x) for x in a.levels.split()):
        ok, wall = run_level(page, dest, ids, a.gen, k, a.rounds)
        if not ok:
            print(f"{k:>4} no samples")
            continue
        ttft = sorted(r[0] for r in ok)
        tpot = sorted(r[1] for r in ok)
        ntok = sum(r[2] for r in ok)
        print(f"{k:>4} {len(ok):>4} {ntok/wall:>8.1f} "
              f"{ttft[len(ttft)//2]:>9.0f} {ttft[int(len(ttft)*0.99)]:>9.0f} "
              f"{tpot[len(tpot)//2]:>9.1f}")


if __name__ == "__main__":
    main()
