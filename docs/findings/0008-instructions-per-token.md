# Finding 0008 — CPU instructions per generated token

Status: measured on 2026-10-08, same host as `bench/results/HOST.md`.
Raw log: `bench/results/insn_token_20261008T065421Z.txt`.

The goal asks for **CPU instructions/token**. This counts the resident
executor's retired instructions with `perf_event_open` (`tools/perfcount.c`,
because the distro `perf` wrapper is broken for this kernel) while it is driven
into saturation, and divides by the tokens it produced in the same window.

## Method

- `tools/perfcount <pid> <ms>` opens `PERF_COUNT_HW_INSTRUCTIONS` and
  `PERF_COUNT_HW_CPU_CYCLES` on the executor PID (`inherit=1`), enables, sleeps
  the window, reads, and prints totals.
- `tools/insn_token.py` drives 4 / 16 concurrent client threads at the executor
  for an 8 s window — **long enough that the executor never goes idle**, so its
  instruction stream is decode work, not poll spin — and counts tokens the
  clients observed produced in exactly that window.
- The executor is host Python (HF `transformers`), so this prices the *whole*
  host token path: Python decode + KV-cache handling + the two `packed_into`
  page writes per token.

## Results

| load | wall | tokens | tok/s | executor insn | insn/token |
|---|---|---|---|---|---|
| 4 threads | 8.12 s | 960 | 118 | 46.264 G | **48.2 M** |
| 16 threads | 8.59 s | 1088 | 127 | 46.719 G | **42.9 M** |

**Sustained executor CPU burn is ~5.7 G instructions/s** (46.3–46.7 G over
8 s) — the executor is pinned near one core for the whole window even though the
GPU does the math. Instructions/token is ~40–50 M, consistent across two load
levels; the spread is the token-count granularity (gen=16 per request), not a
rate change.

## Interpretation

- **The GPU is not the CPU's bottleneck — Python is.** A 0.5 B fp16 model
  decodes at ~8.3 ms/token (120 tok/s, `0005`/`0007`), and the *host* spends
  ~40 M instructions per token driving it. That is ~5.7 G insn/s, i.e. about
  one core saturated per single-stream decode.
- **This is the number a C/CUDA executor removes.** The instruction count is
  dominated by Python's per-token interpreter loop + tensor glue, not by the
  page handoff: the page write is 2 `packed_into` (a few dozen instructions) and
  the poll between requests is a handful of loads. So the ~40 M/token is a
  **host-Python tax**, not a bpf-gpu-server tax — but it is the honest cost of the
  current implementation and the reason the bpfusion path (`0005`) is +0.7 ms
  TTFT over direct: both sit on the same Python decode.
- **Claim boundary:** this is *not* "bpf-gpu-server costs 40 M instructions/token".
  It is "the current host-Python executor costs ~40 M instructions/token,
  measured, with the page path included". The page path alone (one
  `sendto`-equivalent ingress parse + a sub-µs handoff, `0003`/`0004`) is a
  rounding error against it.

## Reproduce

```
make tools
./build/bpfusion_load attach lo
PYTHONPATH=executor python3 executor/llm_executor.py --seconds 90 &
PID=$(pgrep -f llm_executor.py | head -1)
PYTHONPATH=executor python3 tools/insn_token.py --pid $PID --threads 16 --gen 16 --secs 8
```

## Caveats

- `perf_event_paranoid=2` on this host is enough for per-PID counting by root;
  no `-1` needed.
- `inherit=1` counts threads the executor spawns, if any (it spawns none today),
  so no double counting.
- An earlier idle-subtracted variant was **wrong** and is not quoted: the
  executor's poll loop is *more* instruction-dense than decode, so subtracting
  an idle rate exceeded the serving count. Saturation removes the poll entirely
  and is the only valid method here.
