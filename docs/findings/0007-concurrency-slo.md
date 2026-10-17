# Finding 0007 — concurrency and the SLO picture of the current executor

Status: measured on 2026-10-08, same host as `bench/results/HOST.md`.
Raw log: `bench/results/llm_load_20261008T063416Z.txt`.

`0005` measured a single request. This drives **K concurrent requests** at the
resident executor and reports, per concurrency level, aggregate token goodput
and per-request TTFT/TPOT (`tools/llm_load.py`).

## Results (Qwen2.5-0.5B, gen=8, 2 rounds/thread)

| concurrency | requests | tok/s | TTFT p50 | TTFT p99 | TPOT p50 |
|---|---|---|---|---|---|
| 1 | 2 | 117.8 | 9.3 ms | 9.3 ms | 8.4 ms |
| 2 | 4 | 119.0 | 72.3 ms | 75.9 ms | 8.8 ms |
| 4 | 8 | 118.7 | 138.2 ms | 206.5 ms | 9.4 ms |
| 8 | 16 | 118.8 | 274.9 ms | 402.2 ms | 8.4 ms |

## Reading

- **Token goodput is flat at ~119 tok/s** regardless of concurrency. The
  executor serves **one request at a time** (batch 1), so extra concurrency
  buys no throughput — it only enlarges the queue.
- **TTFT grows ~linearly with the queue depth**: 9.3 ms (empty) → 72 ms → 138 ms
  → 275 ms as concurrency goes 1 → 2 → 4 → 8. That is `(position) × per-request
  generation time` — a new request waits for every request ahead of it to finish
  all 8 tokens.
- **TPOT is essentially constant (~8.4 ms)**: once a request starts, its decode
  runs at the single-stream rate; queueing delays the start, not the per-token
  pace.

This is the honest SLO picture of the **current** resident design: low-latency
for the first request, unbounded TTFT tail under load, no throughput scaling.
The fix is continuous/batched serving (share a forward pass across requests),
which is not implemented — the executor runs one HF `generate` loop per slot.

## Bugs found and fixed while measuring

- **Executor livelock:** on a slot whose `state != PENDING` the loop did
  `continue` without advancing `seen`, so it spun forever and stopped serving.
  Fixed: advance `seen` past skipped slots; also resync `seen = head - BF_LLM_SLOTS`
  when more requests are published than the 8-slot ring holds.
- **Concurrent client slot-search race:** matching on `client_ns` **and**
  `state == PENDING` missed a slot the executor had already recycled before the
  client looked, hanging the client until its timeout. Match on `client_ns`
  alone (the tag survives slot reuse).

Both only surface under concurrency; the single-request path (`0005`) hits
neither.

## Reproduce

```
make
./build/bpfusion_load attach lo
PYTHONPATH=executor python3 executor/llm_executor.py --seconds 60 &
PYTHONPATH=executor python3 tools/llm_load.py --levels "1 2 4 8" --rounds 2 --gen 8
```

## What this implies for the goal

The bpf-gpu-server claim is about the **queue path**, not the model server. This
measurement quantifies how much headroom the queue path leaves for a real
server: at 119 tok/s decode the ~0.7 ms queue increment (`0005`) is 8 % of one
token of latency, and the page round trip (<100 µs, `0003`/`0004`) is ~1 % — so
the transport is not the bottleneck the way a per-request `sendmsg`/wake would
be. Making this a *serving* path needs continuous batching in the executor.
