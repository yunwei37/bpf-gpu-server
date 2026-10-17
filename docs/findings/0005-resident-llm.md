# Finding 0005 — real Qwen LLM on the pinned page (host-Python executor)

Status: measured on 2026-10-08, same host as `bench/results/HOST.md`.
Raw log: `bench/results/llm_20261008T055510Z.txt`.

This is the first **real model** on the bpf-gpu-server path from the goal
(Qwen 0.6B/1.7B or a tiny MLP). The MLP path in `0004` proved the queue
mechanics; this proves the request and generated tokens travel through the
kernel→page path with a transformer.

## What runs

Model: **Qwen2.5-0.5B-Instruct**, fp16, HuggingFace `transformers`, batch 1.

```
client sendto (UDP :39402)  →  tc/clsact ingress eBPF
     parse magic + n_prompt + n_gen + client_ns, copy n_prompt token ids
     into page.llm[idx].tok_in, publish state=PENDING, llm_head++
        |
        v
resident executor (executor/llm_executor.py)
     polls page.llm_head; on a new PENDING slot runs prefill + decode and
     writes each generated token id to page.llm[idx].tok_out[k], then
     bumps slot.produced (release) so it becomes visible
        |
        v
client polls the same slot; TTFT = sendto → produced >= 1,
TPOT = inter-token gap between successive `produced` bumps
```

The **request tokens and every generated token pass through the shared
pinned BPF page**. There is no socket read path on the response side and no
side channel: the page *is* the channel, exactly as for the MLP path.

The page gained an independent token ring (`bf_llm_slot llm[8]`,
`llm_head`) at the **tail** of `struct bf_page`, so the MLP-field offsets are
unchanged. Both rings are owned by the same producer/consumer discipline.

## Measurement

Matched paths, same model, same host, same process (the executor is host
Python; the direct baseline is the same model in the same process):

| path | TTFT p50 | TPOT p50 |
|---|---|---|
| direct (in-process `model.generate`) | 9 035 µs | 8 391 µs |
| **bpfusion** (packet → page → ggpu → page) | **9 739 µs** | **8 955 µs** |

Increment of the full bpfusion path over direct:

- **TTFT: +704 µs (+7.8 %)**
- **TPOT: +565 µs (+6.7 %)**

That increment is the entire cost of routing the request through the `tc`
hook into the page, the executor noticing it, and the client observing the
first token — with a 0.5 B model whose own prefill is ~9 ms. It is a small
fraction because the model dominates.

## Honest scope

- **The absolute numbers are not a serving claim.** The executor is plain
  HF `transformers`, batch 1, Python, one request at a time. It is *not* a
  vLLM/megakernel. The comparison is only valid as a **relative** measure of
  the queue-path increment, because both sides run the identical model.
- **The transport on the response side is a host poll, not a GPU write
  path.** The GPU runs the model; the tokens are written into the page by the
  Python executor. What is kernel-native is the *ingress* (eBPF `tc` program
  copies prompt tokens into the page) and the fact that both directions share
  one mmap'ed page with no copy or socket syscall in the model loop.
- **Python poll overhead is in the increment.** The page round trip itself
  measured sub-100 µs on the MLP path (`0003`/`0004`); the ~0.7 ms added here
  is dominated by the Python interpreter noticing `llm_head`/`produced`
  (GIL-bound polling). A C or CUDA executor would shrink it.
- **Payload cap.** 64 prompt tokens + 64 generated tokens per request
  (`BF_LLM_MAX_TOK`), sized to the page slot. Longer prompts need a larger
  slot or a staging buffer.
- **Loopback only.** The `tc` hook is on `lo`.

## Reproduce

```
make
./build/bpfusion_load attach lo
PYTHONPATH=executor python3 executor/llm_executor.py --seconds 90 &
PYTHONPATH=executor python3 tools/llm_bench.py --gen 32 --rounds 8
# or: ./bench/run_llm.sh 90 32 8
```

Sample output (`bpfusion` path, gen=16): `' Paris. It is the largest city in
Europe and the third largest city in the'` — a correct continuation of
`The capital of France is`.
