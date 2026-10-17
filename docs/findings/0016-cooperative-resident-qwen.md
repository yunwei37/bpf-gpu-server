# 0016: cooperative resident Qwen and the longer-decode accuracy boundary

Observed 2026-10-10 UTC in the existing bpf-gpu-server Workspace, RTX 5090,
CUDA 13, Linux 7.3-rc3, Qwen2.5-0.5B-Instruct, eager HF fp16 oracle.

## Initial scalar grid: one launch

The real-model reference now uses one resident cooperative CUDA grid instead
of one CTA. The launcher discovers the device SM count (170 on this GPU),
checks cooperative-launch support and that the kernel can reside with at
least one 256-thread CTA per SM, then launches once. There is no per-request
host launch, copy, sampling, accept or send loop.

Matrix rows and elementwise operations are distributed across the grid;
warp dot-product order and fp16 rounding boundaries remain the same as the
single-CTA reference. Attention heads are assigned to global warps. RMSNorm
and vocabulary argmax retain their original CTA-local reductions in CTA zero.
Grid barriers publish intermediate device buffers. CTA zero publishes queue
head/state, request dimensions and selected token to a shared device control
buffer, so all CTAs take the same branches and reach the same barriers.
Polling the mapped queue independently in each CTA would make barrier
participation nonuniform and could deadlock.

This follows the [CUDA cooperative groups synchronization contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html):
all participating threads must reach a barrier, which makes preceding memory
accesses visible within the group. The grid must be launched cooperatively
and fit the device's residency constraints. There is no hand-built inter-CTA
spin barrier or model-specific SM-count override.

## Passing scope

```sh
make probes tools qwen
make -C module CC=gcc-15
python3 tests/resident_qwen.py
```

[The grid log](../../bench/results/resident_qwen_grid_20261010.txt) passes the
existing strict oracle: 26 requests / 208 generated token IDs, four prompts,
12 binary and 14 HTTP requests, eight concurrent clients, six pipelined
requests across ring wrap, byte-separated writes and a half-closed stream.
Kernel counters show 26 completions, 1,938 reply bytes, 5,021 accepted/closed
connections, zero abandonment, TX retries, registration failures and drops.
Strace follows all executor threads and finds no accept/receive/send calls
after ready. Module unload and BPF detach complete at shutdown. The test now removes its
three control/statistics/doorbell map pins as well, including when an oracle
assertion fails; previously only the stream link/map were removed.

After a storage-pressure eviction during baseline preparation, the same PVC and
Workspace were restored and the executor was rebuilt. The
[post-recovery run](../../bench/results/resident_qwen_grid_recovery_20261010.txt)
again passes all 26 requests / 208 tokens against the same PyTorch 2.14.1+cu130
and Transformers 5.19.0 versions. This is a workload recovery receipt, not
a new performance comparison. The
[cleanup regression](../../bench/results/resident_qwen_grid_cleanup_20261010.txt)
also passes the strict oracle and confirms removal of the remaining map pins.

Request wall times in this correctness test include one millisecond sleeps
between transmitted bytes. They are not an inference-only latency measure
or a matched serving comparison. No serving speedup or total-CPU saving is
claimed from this log.

## Failing scope and diagnostic control

The test now accepts `--gen` (within the existing queue's 1..64 output bound)
and `--executor` to exercise another binary with the same strict oracle.

```sh
python3 tests/resident_qwen.py --gen 64
python3 tests/resident_qwen.py --gen 64 --executor /tmp/bpfusion-qwen-single
```

The [64-token grid run](../../bench/results/resident_qwen_grid64_20261010.txt)
passes the first prompt but fails the second, `What is 2 plus 2?`, at output
position 21 (one-based). A control binary compiled from the previous main
commit `5302639a6826ad3311b951925e8112100d607c93` produces the **same 64 CUDA
token IDs**, including that divergence:
[the single-CTA control](../../bench/results/resident_qwen_single64_20261010.txt).
This failure was already present in the model reference; distributing work
across CTAs did not introduce it in this control.

A separate diagnostic teacher-forced the shared prompt plus the first 20 HF
tokens, generated one token, and copied the final logits only after stopping
the kernel. [Its log](../../bench/results/resident_qwen_logits_20261010.txt)
shows the decisive near tie:

| Token ID / text | HF fp16 logit | CUDA reference logit |
|---|---:|---:|
| 5145 / ` Start` | 20.671875 | 20.65625 |
| 64547 / ` Identify` | 20.671875 | 20.671875 |

HF argmax breaks the tie toward the lower ID, 5145; CUDA selects 64547.
The vocabulary-wide maximum absolute logit error was 0.044921875 and mean
absolute error 0.0071140439 at that prefix. Different floating-point reductions
are a plausible explanation, consistent with [PyTorch numerical accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
but this experiment does not locate every numerical difference or establish
broader model quality. No tie bias, relaxed token assertion or replacement
oracle was added to conceal the failure.

Thus the verified exact-token scope remains the recorded eight-token cases.
The 64-token oracle is explicitly **failed**; general long-decode bitwise
agreement with HF is not claimed. At this initial checkpoint the reference used scalar fp32
accumulation with fp16 boundaries, without tensor-core GEMM or continuous batching.
GPU text processing, robust malformed-request/error handling, completion-event
measurement, real-NIC controls and a matched serving comparison remain open.

## Tensor Core follow-up, 2026-10-10 UTC

The following runs start from `26d207e7a2ca376e0d02a01d4fa35be39ff01eec`
plus the source change recorded with these results. Linear layers now use
16x16x16 WMMA with fp16 inputs and fp32 accumulation. One CTA computes
sixteen output rows; its eight warps divide the K tiles and reduce their
partial sums in fp32 before the existing fp16 output/bias boundary. The
input vector is repeated in B's columns; output column zero is the matvec.
The old scalar calculation remains for untiled dimensions or unaligned
weights. Warp-private shared tiles are 32-byte aligned; warp, CTA and grid
barriers retain uniform participation. See the [CUDA WMMA contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).
This is still one sequence at a time, not continuous batching.

RoPE inverse frequencies are initialized once on the CPU as reciprocal
positive powers, matching the installed Transformers 5.19.0 Qwen2 initializer.
[Build and initialization evidence](../../bench/results/resident_qwen_wmma_build_20261010.txt)
records exact agreement of all 32 fp32 frequencies for head dimension 64,
theta 1000000, with PyTorch 2.14.1+cu130. The unchanged binary model export
is consumed directly; CPU initialization adds no steady request worker.

```sh
make qwen qwen-control
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --executor ./build/qwen_host_launch
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64
```

The final split-K candidate's [resident regression](../../bench/results/resident_qwen_wmma_splitk_resident8_20261010.txt)
and [host-dispatch regression](../../bench/results/resident_qwen_wmma_splitk_host8_20261010.txt)
each pass 285 valid TCP/HTTP requests, 2280 exact oracle tokens and 259
rejections. Each also exercises 5000 empty connections, eight concurrent
clients, 256 rejection/valid pairs across ring wrap, six ordered pipelined
responses, half-close and invalid token IDs. Their launch counts are 1 and
544 respectively. Executor-thread traces find no accept/receive/send after
ready; the host-dispatch control still has its request launch worker.
The [default Makefile binary regression](../../bench/results/resident_qwen_wmma_default8_20261010.txt)
repeats the resident pass. Build evidence shows identical device SASS for
resident/control and for default/candidate builds: 96 registers/thread,
32-byte stack, 15396-byte shared storage and 30 static tensor instructions.
These are build/correctness receipts, not throughput measurements.

The [final 64-token run](../../bench/results/resident_qwen_wmma_splitk64_20261010.txt)
still **fails**. The first two prompts match all 64 outputs; the third,
`Write a short greeting.`, matches its first 62 outputs then selects token
11 (`,`), whereas HF selects 323 (` and`). The fourth prompt is not reached
by the fail-fast test. This extends the observed matching prefix relative
to the scalar checkpoint's second-prompt divergence at 21; it does not
establish general long-output correctness. Neither oracle nor tie-breaking
was weakened.

For provenance, the first WMMA candidate used one warp per sixteen rows
with the whole K sum. Its [eight-token resident](../../bench/results/resident_qwen_wmma8_20261010.txt)
and [host-dispatch](../../bench/results/resident_qwen_wmma_host8_20261010.txt)
regressions passed, but its [64-token run](../../bench/results/resident_qwen_wmma64_20261010.txt)
failed at the same third-prompt position 63. Adding the CPU frequency table
[alone](../../bench/results/resident_qwen_wmma_rope64_20261010.txt) preserved
those first-three-prompt outputs. The final split-K calculation does too.
The first pre-recovery attempt was interrupted by a storage eviction and
has no oracle verdict; the logs above were collected after recovery on
the original PVC and same library versions.

A [post-stop diagnostic of the first WMMA candidate](../../bench/results/resident_qwen_wmma_logits_20261010.txt)
asserts the identical 62-token prefix, then copies final hidden state,
normalized state and logits only after stopping the resident kernel. Final
residual max absolute error is 0.03125; final norm max error is 0.1875.
Applying PyTorch's output linear layer to the *CUDA normalized state* also
selects token 11. Same-input output-layer max logit error is 0.0078125,
compared with 0.06640625 end-to-end. At the decisive tie, HF has logits
24.5 for 323 and 24.484375 for 11; CUDA has 24.5 for both. Upstream hidden
state differences therefore matter at this prefix. This diagnostic does
not locate the first differing layer or prove all errors have one cause.
The temporary snapshot code is absent from the default executor.

No matched serving-performance comparison has been run for this changed
math. Finding 0018's measurements and identical-SASS claims refer to their
recorded earlier commits, not this new device kernel. Correct long decode,
GPU text processing, continuous batching, completion-event measurement,
real-NIC controls and optimized serving comparisons remain open.

## Layer diagnosis and matrix prefill, 2026-10-10 UTC

[Layer snapshots](../../bench/results/resident_qwen_layers_20261010.txt)
compare source `75567807b3ccc2bd00f63f5643a9a05f13ade9bb` with the same HF
eager fp16 oracle at first input position 0 and final forward position 66
for `Write a short greeting.`. A temporary native binary saves 18 stages
per layer on-device and copies them after stopping. HF forward hooks save
the corresponding stages; its attention wrapper returns the original
attention result. The instrumented native run retains the identical first
62 outputs and the known output-63 divergence. This is a numerical probe,
not a serving timing or no-worker regression.

At position 0, layer-zero embedding and input RMSNorm agree exactly;
Q projection differs at one element by 0.00000095367431640625. The
post-attention residual and normalization agree, but gate/up projections
differ at 21/25 elements, and the layer output differs at 295/896 elements.
At position 66 the same layer's embedding, input norm and Q/K/V projections
agree; its output differs at 69/896 elements. At layer 23, the final output
max absolute difference is 0.0625. These locate early observed projection
rounding differences and their downstream accumulation at two positions;
they do not establish one operation as the cause of all later differences.

Same-input controls use layer-zero weights with the saved CUDA input.
At position 0, the gate projection differs from CPU fp64-dot-to-fp16
rounding at 3 rows; single-token PyTorch differs at 5 rows; the two differ
at 8 rows. HF's actual multi-token prefill differs from CUDA at 21 rows.
Thus neither higher precision nor matching the abstract linear operation
alone guarantees the recorded HF result. An [fp64 linear-accumulation
candidate](../../bench/results/resident_qwen_fp64_control64_20261010.txt)
fails the first prompt at output 14. It is not retained as a fix. No oracle,
tie rule or precision assertion was relaxed.

The current executor processes a whole prompt through each layer: WMMA
B columns carry its token vectors, per-token CTA reductions normalize
vectors, token/head warps perform causal attention, and all prompt K/V
entries are published before attention. Decode still uses one token.
Scratch space follows the existing 64-token input bound. Each normalization
CTA strides across tokens, so correctness does not require at least 64 SMs.
Model weights, fp16 boundaries, greedy sampling, queue ownership and kernel
TX are unchanged. Requests remain serial; this is prompt prefill batching,
not continuous multi-request batching.

```sh
make qwen qwen-control
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --prefill-edges
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --prefill-edges --executor ./build/qwen_host_launch
```

The new optional test covers 16/17/32/64-token prompts by repeating the
first existing prompt's token IDs, including tile boundaries and maximum
input length. Its [resident](../../bench/results/resident_qwen_prefill_edges_resident8_20261010.txt)
and [host-dispatch](../../bench/results/resident_qwen_prefill_edges_host8_20261010.txt)
runs each pass 297 valid TCP/HTTP requests, 2376 exact oracle tokens and
259 rejections with the existing churn, concurrency, ring-wrap, pipeline,
half-close and trace checks. Launch counts are 1/556.
[Default build evidence](../../bench/results/resident_qwen_prefill_build_20261010.txt)
confirms identical resident/control and default/candidate device SASS,
with 96 registers/thread, 32-byte stack and 15396-byte shared storage.

The [first matrix-prefill 64-token candidate](../../bench/results/resident_qwen_prefill64_20261010.txt)
and [final default 64-token binary](../../bench/results/resident_qwen_prefill_default64_20261010.txt)
still fail at the third prompt's output 63, retaining the recorded first-three-
prompt outputs. Batching prefill within the custom WMMA implementation alone
did not resolve the HF reduction difference. No matched performance benefit
is claimed for this increment. Full long-output correctness, text processing,
continuous batching, completion-event/NIC measurement and optimized baselines
remain open.
