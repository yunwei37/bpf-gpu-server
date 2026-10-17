#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: LLM-path client.
#
# Sends a prompt (token ids) as a UDP datagram to BGS_LLM_PORT through the tc
# ingress hook, then watches the pinned page's LLM ring for the executor's
# generated tokens. TTFT is measured from sendto to the first token appearing in
# the page; TPOT is the inter-token gap. Both cross the real kernel->page path.
import argparse, ctypes, mmap, os, socket, struct, sys, time

from llm_executor import (LLM_RING_OFF, LLM_SLOT_FMT, LLM_SLOT_SIZE,
                          BGS_LLM_SLOTS, LLM_HEAD_OFF, open_page, Page)

BGS_LLM_PORT = 39402
BGS_LLM_MAGIC = 0x514c4d51


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--port", type=int, default=BGS_LLM_PORT)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(a.prompt).input_ids[:64]
    n_prompt = len(ids)
    assert n_prompt > 0

    mm, base = open_page()
    page = Page(mm, base)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", a.port)

    ttfts, tpots = [], []
    for r in range(a.rounds):
        client_ns = time.monotonic_ns()
        hdr = struct.pack("<IIIIQ", BGS_LLM_MAGIC, n_prompt, a.gen, 0, client_ns)
        payload = hdr + struct.pack(f"<{n_prompt}I", *ids)
        t_send = time.perf_counter()
        s.sendto(payload, dest)

        # Locate our slot: scan the ring for a PENDING/populated slot whose
        # client_ns matches (the executor publishes into the same page).
        deadline = t_send + 30.0
        slot = None
        while time.perf_counter() < deadline:
            head = page.u32(LLM_HEAD_OFF)
            for k in range(BGS_LLM_SLOTS):
                idx = (head - 1 - k) % BGS_LLM_SLOTS
                v = page.slot(idx)
                if v[7] == client_ns:  # client_ns field
                    slot = (idx, v)
                    break
            if slot is not None:
                break
        if slot is None:
            print(f"round {r}: no slot found (dropped?)")
            continue
        idx = slot[0]

        t_first = None
        last_produced = 0
        gens = []
        while True:
            v = page.slot(idx)
            produced = v[3]
            if produced > last_produced:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                while last_produced < produced:
                    gens.append(v[11][last_produced])
                    last_produced += 1
                if produced >= a.gen:
                    break
            if time.perf_counter() > deadline:
                break
        if t_first is None:
            print(f"round {r}: no tokens")
            continue
        ttft = (t_first - t_send) * 1e6
        # token timestamps: we only recorded the first; approximate TPOT from
        # total decode span we can recompute by re-polling is not stored, so
        # measure TPOT as (last-observed-first)/(n-1) using a follow-up scan.
        ttfts.append(ttft)
        text = tok.decode(gens)
        print(f"round {r}: prompt={n_prompt} tok gen={len(gens)} "
              f"TTFT={ttft:.0f} us text={text!r}")

    if ttfts:
        ttfts.sort()
        print(f"TTFT us: p50={ttfts[len(ttfts)//2]:.0f} "
              f"min={ttfts[0]:.0f} max={ttfts[-1]:.0f}")


if __name__ == "__main__":
    main()
