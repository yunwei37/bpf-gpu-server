# 0021: real-Qwen compute diagnostics and last-position logits

Finding [0020](0020-graph-dispatch.md) completes the whole-request dispatch
ablation. It does not detect a broad TTFT advantage and exposes serial
queueing at eight clients. The next engineering work targets actual computation
and batching while preserving the unchanged independent eager-fp16 oracle.
This diagnostic is not a new paper experiment or competitive serving result.

## Reproduction and scope

The raw profile files include the exact temporary diagnostic Python script
and command. The capture source in these pairs is the pre-optimization
`779e2f3` version (also unchanged from `e3f272e`). To reproduce against a later
checkout without changing main, load it with
`git show 779e2f3:executor/qwen_graph.py` in the diagnostic instead of reading
the current file; retain the documented native graph library and dependencies. It reads the original graph-capture source, captures all 190
model graphs, and replaces the native serve call with ordinary host graph
replays. The original exported real weights, HF operations, cache construction,
copy-back and device argmax are retained. Actual prompt token IDs populate the
input buffer; decode prefixes are prepared by the preceding model graphs.
Five sampled shapes cover prefill lengths 5/64 and decode positions 16/64/126.
Each has ten warm replays, thirty CUDA-event samples and a separate
[PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html) replay.

These timings exclude TCP, GPU-tail scheduling and kernel TX. Samples are
correlated repetitions in one process per variant, run sequentially rather
than paired/randomized process blocks. GPU clocks/power were not integrated
for these profiles. Do not turn their differences into a causal serving
speedup, stable tail estimate or energy result. Profiler event counts describe
CUDA device events, including any device copies, rather than CPU launches.

## SDPA candidate rejected by the fixed oracle

The bounded candidate used official
[HF SDPA](https://huggingface.co/docs/transformers/en/attention_interface)
and `logits_to_keep=1`. The
[eager profile](../../bench/results/qwen_compute_eager_profile_20261010.txt)
and [candidate profile](../../bench/results/qwen_compute_sdpa_profile_20261010.txt)
record decode-position64 event means of 2.7433 and 2.5625 ms and 1,245/1,077
CUDA device events. These are diagnostic observations, not an accepted
end-to-end improvement.

The [combined candidate regression](../../bench/results/qwen_sdpa_last_logits_correctness_20261010.txt)
failed the first 64-token prompt. Isolating changes showed that
[SDPA alone](../../bench/results/qwen_sdpa_only_correctness_20261010.txt)
also failed: output14 (zero-based13) was 3283 instead of oracle304.
It completed the TCP reply with graph_error=0 and clean teardown; this was
an output mismatch, not admission failure. The exact source delta was only
`attn_implementation='eager'` to `'sdpa'`; the combined candidate also used
last-position logits. No SDPA change is retained, and no oracle or precision
criterion was weakened. This failure bounds the tested backend against the
chosen exact oracle; it does not prove SDPA is generally incorrect.

## Retained change: compute only the needed logits

The actual HF forward supports `logits_to_keep`, slicing hidden states before
its vocabulary projection. The executor consumes only the last-position
argmax. The sole production change passes `logits_to_keep=1` to that forward;
attention remains eager, KV/precision/dispatcher/protocol are unchanged.

The isolated
[original profile](../../bench/results/qwen_eager_last_logits_profile_20261010.txt)
and [last-position profile](../../bench/results/qwen_last_logits_last_logits_profile_20261010.txt)
record the following post-capture PyTorch counters:

| capture variant | CUDA allocated bytes | CUDA reserved bytes | held logits bytes |
|---|---:|---:|---:|
| all positions | 1,839,313,920 | 2,669,674,496 | 670,341,632 |
| last position | 1,218,157,568 | 1,537,212,416 | 57,735,680 |

Allocated memory falls by **621,156,352 bytes**, about 0.62 GB. These counters
cover PyTorch-owned allocations at the same capture phase, not total GPU or
native device-graph memory. The held-logits reduction is expected from keeping
190 final-position rows rather than 2,206 rows across the catalogue.
This removes unused outputs and supplies a material memory benefit for future
batching; it is not evidence that batching already runs.

The isolated event timings were 2.5981/2.6054 ms for prefill5,
3.1337/3.1175 ms for prefill64 and 2.7400/2.7408 ms for decode-position64
(original/last-position). No latency benefit is claimed from these observations.
Finding0020's measured serving numbers precede this source change; they are
not relabeled as fresh measurements of the optimized version.

Both complete strict gen64 candidate regressions passed the original oracle:
[resident](../../bench/results/qwen_last_logits_correctness_20261010.txt) and
[host-launch](../../bench/results/qwen_last_logits_host_correctness_20261010.txt),
**303 valid requests / 19,392 exact tokens / 259 rejections per mode**,
5,000 empty connections, eight concurrent clients, reject-to-valid reuse,
pipeline/ring wrap/half-close, zero admission drops, graph_error=0, expected
host/model launch counts, all-thread network-syscall checks and clean module/pin
teardown. The temporary candidate differed from the retained script by exactly
this forward argument and used the matched native libraries from0020.

## Default RMSNorm fusion probe

The next bounded candidate compiled only the existing `Qwen2RMSNorm.forward`
with default `torch.compile`, leaving eager attention and the original model
operations around it. Its
[real gen64 regression](../../bench/results/qwen_compiled_norm_default_correctness_20261010.txt)
reached the server and returned all64 tokens with graph_error=0, but failed
at output14 of the first prompt, also3283 instead of oracle304. Network trace
and clean module/pin teardown passed. This is a failed default candidate, not
an accepted performance result. No compiler change is retained.

The installed PyTorch2.14.1 Inductor source documents that fusion may remove
fp16 downcast/upcast boundaries and provides `emulate_precision_casts` to
preserve them. That is a concrete reason to test this existing compiler option
in the temporary candidate after the default failure. It is not proof that
cast removal caused this output mismatch, nor permission to change the oracle.
The [precision-preserving regression](../../bench/results/qwen_compiled_norm_precision_correctness_20261010.txt)
also completed all64 tokens but failed at the same output14 (3283 instead of
304), with graph_error=0, passing network trace and clean module/pin teardown.
The default candidate used the pre-last-position-logits source; this second
candidate used the retained last-position argument and
`torch.compile(Qwen2RMSNorm.forward, options={"emulate_precision_casts": True})`.
Neither compiler candidate is retained. These failures do not identify which
change in arithmetic caused the divergence, and no speedup is claimed.

The full serving goal remains active. Reduce the remaining pointwise/cache
work and serial queueing, measure an accepted compute improvement on real TCP,
and complete text, optimized serving, total-host-cost and physical-NIC evidence.


## New pointwise candidate: preserve RoPE casts while fusing operations

The complete concurrent-service comparison in finding0023 establishes a local
throughput/queueing improvement, with slower per-request TPOT and inconclusive
single-client effects. Existing profiles show many small pointwise/copy events
in each decode graph. RMSNorm and SDPA accuracy failures above remain closed;
this candidate instead compiles only the existing HF apply_rotary_pos_emb.
Its operations rotate/negate, multiply and add without a reduction or a
projection/batch-shape change. torch.compile uses the default backend/mode,
with emulate_precision_casts=True to preserve the existing fp16 boundaries.
No oracle, norm, attention, projection, KV, graph-bank ownership, native
scheduler, page/module or transport changes are included.

[Exact temporary candidate source/delta](../../bench/results/qwen_compiled_rope_candidate_20261010.txt)
records the two added capture-script lines at64b1e7e. At that qualification
milestone production source remained unchanged pending default-native
qualification and a matched serving result; the completed decision is below.
The [first GPU-only attempt](../../bench/results/qwen_compiled_rope_qualification_20261010.txt)
failed before candidate load/capture: the reused old probe library does not
export the new service max_requests symbol. The wrapper already discovers its
two banks; removing that unused restype lookup is the only harness repair.
The [completed qualification](../../bench/results/qwen_compiled_rope_qualified_20261010.txt)
passes all12 two-bank groups/1,536 exact tokens, including ten gen64
prompt-edge cases and two mixed groups, with one host launch/128 model graphs/
error0 per group. This is GPU-controlled computation qualification, not a TCP
serving result or proof of general numerical equivalence.

Sequential one-process CUDA-event/profiler diagnostics preserve the exact
scripts: [baseline](../../bench/results/qwen_baseline_rope_profile_20261010.txt)
and [RoPE candidate](../../bench/results/qwen_rope_rope_profile_20261010.txt).
Each captures one bank with the same shared KV and samples five graph shapes
using ten warm/thirty measured host replays. They exclude TCP, native device
instantiation and GPU-tail scheduling. CUDA device events decrease by192 at
every sampled shape: prefill5 1197 ->1005, prefill64 1269 ->1077, decode
positions16/64/126 1149 ->957. Torch allocation/reservation/held logits are
unchanged at1,089,993,728 /1,518,338,048 /57,735,680 bytes. Event times are
sequential correlated diagnostics on a shared GPU; their differences do not
establish causal compute or serving speedup. A real paired service comparison
is necessary to decide whether to retain this candidate.

## RoPE serving comparison plan

This is supporting engineering evidence for the user's throughput/latency
objective; there is no paper RQ/thesis to invent. The unresolved question is
whether reducing pointwise model-graph work improves actual TPOT/throughput
with the qualified no-userspace-worker concurrent serving mechanism. Positive
matched effects justify retaining the small exact candidate; negative or
inconclusive effects leave the production capture unchanged and direct work
elsewhere. Finding0023 changed concurrency, not this model computation, and
its results cannot settle the RoPE effect. Repeating KV/dispatch or the closed
norm/SDPA candidates would add no answer. This is the strongest immediately
runnable component comparison; optimized serving remains a separate wider
service evaluation, not a condition for publishing this local result.

Baseline is the current concurrent private-bank capture at64b1e7e. Candidate
changes only the two qualified RoPE compile lines. Both discover eight banks,
share the identical native library (with the same checked temporary listener
TCP_NODELAY call), and use the same weights, model shapes/eager fp16 operations
outside RoPE, shared KV, independent batch1 oracle, GPU sampling, FIFO ring,
module and one host service launch. This is a current-practice/component
baseline, not a SOTA/optimized external server. The serial host control only
provides correctness coverage. [Adapter/raw source provenance](../../bench/results/qwen_compiled_rope_adapter_20261010.txt)
contains the exact renamed existing driver, common native source diff,
symlink targets, frozen capture-source delta and instantiated commands.

Before full collection, complete both default-native strict gen64/prompt-edge
TCP regressions and the real preflight. Keep every original oracle assertion,
request/shutdown timeout, trace/lifecycle check and metric; readiness remains
ready-or-process-exit as already qualified for bank capture. Use the owning
ephemeral compiler cache, without a permanent helper or configuration layer.
No candidate production change occurs before the complete comparison and
result review.

The fixed matrix is four real prompts, gen1/gen64, clients1/8, five alternating
paired process blocks with rotating matching cell order,64 requests/cell
(16 per prompt),eight excluded warmups/process and a separate one-second
idle CPU sample. Full result is40 cells,2,560 exact requests,83,200 measured
tokens and ten normal exits; each process must have one service launch,
8,580 device model graphs/error0,264 publications,zero admission/registration
errors,eight FREE slots,and clean module/hooks/pins. Default tests additionally
cover invalid input, mixed lengths, persistent reuse, pipeline and half-close.
The separate real preflight uses gen1/client1/four requests per engine and
establishes executability only.

Primary metrics are gen64 TPOT and total output TPS at clients1 and8,
following [NVIDIA definitions](https://docs.nvidia.com/nim/benchmarking/llm/metrics).
TPOT=(last-first)/63; TPS is all output tokens divided by first-send to
last-arrival. Assess all four primary effects separately using all five paired
blocks and nominal two-sided t95,df4,without familywise correction. Positive
means baseline-minus-RoPE TPOT or RoPE-minus-baseline TPS. A positive one and
inconclusive others is mixed support; report contradictions and tradeoffs
without selecting cells post-hoc or collapsing to a weighted score. TTFT,
completion,p99 descriptions,gen1,bootstrap and executor-PID/idle CPU remain
reported diagnostics. Supplementary completion analysis reuses stored
request timestamps; the barrier/future CPU window is intentionally broader
than the TPS denominator. Connection/thread creation and model/capture
bootstrap precede active measurement; record bootstrap separately.

Record GPU endpoint clocks/temperature/power/utilization and host residency
before/after; preserve the SGLang co-tenant. These coarse snapshots do not
prove isolation,within-window overlap,integrated energy or steady co-tenant
load. No fixed isolated percentage,physical-NIC,open-load,total-host/energy
or competitive-serving claim follows. Fresh read-only plan/result reviewers
assess validity and the declared narrow decision; no separate saved report is
claimed. Exact full/preflight commands and raw-result paths are in the adapter
raw. All sources/driver remain frozen during collection.


Default-native candidate qualification completed in both modes:
[resident](../../bench/results/qwen_compiled_rope_resident_correctness_20261010.txt)
and [serial host control](../../bench/results/qwen_compiled_rope_host_correctness_20261010.txt),
each335 valid requests /20,432 exact tokens /259 rejects,5,000 empty
connections, eight-client persistent invalid-to-valid reuse,32 mixed-length
requests,ordered pipeline,half-close,ring wrap,zero errors,eight FREE slots,
all-thread network trace and clean module/hooks/pins. Resident has one host
launch,banks8,max_active8; host control has594 host launches,banks1,max_active1.
The capture candidate remains temporary; this proves correctness at default
transport,not serving benefit.

Fresh read-only plan review admitted the frozen two-line candidate and matched
comparison, conditional on both strict qualifiers and real preflight passing.
Both qualifiers are now terminal and pass. Reviewer independently confirmed
source identity/common binary,independent oracle,all12 GPU groups,count/sign/
metric formulas,new decision value and minimal source growth. Preserve all
four primary effects and the shared-GPU/nominal-interval boundary; report
bootstrap separately with the already populated ephemeral compiler cache.
Result review must independently verify counters/FREE/cleanup. No separate
saved reviewer report is claimed.


Real preflight completed normally on the frozen source64b1e7e: [raw](../../bench/results/qwen_compiled_rope_preflight_20261010.jsonl)
and [stdout](../../bench/results/qwen_compiled_rope_preflight_20261010.stdout.txt),
15 records/eight exact measured gen1 requests/two normal exits. Each engine
records12 publications,264 model graphs,one host service launch,error0,
banks8/max_active1,zero queue/registration errors and eight FREE slots.
Module and pins are absent after cleanup. The compiler cache was already
populated by qualification/profile. All plan-review conditions are closed;
preflight timings are setup evidence and are not a serving result. Main58beb1d
is the doc-only qualification milestone; its production source is unchanged
from64b1e7e. Full collection keeps both frozen capture scripts,identical
native library and exact driver unchanged.


## Complete RoPE TCP result

The [full raw](../../bench/results/qwen_compiled_rope_20261010.jsonl) and
[stdout](../../bench/results/qwen_compiled_rope_20261010.stdout.txt) terminate
normally with EXIT0 at 58beb1d: all 40 cells,2,560 requests,83,200 exact output
tokens and ten normal executor exits. No full-run failure or exclusion occurred.
Every process records one host service launch,8,580 device model graphs,
graph_error=0, banks8/max_active8,264 publications, zero queue/registration errors
and eight final FREE slots. Both frozen scripts use the same native binary;
the candidate changes only the qualified RoPE helper.

The [supplementary analysis](../../bench/results/qwen_compiled_rope_analysis_20261010.txt)
contains its exact script and independently checks every oracle/timestamp
metric, TPS first-send/last-arrival denominator, prompt balance, alternating process
order, counters and all five paired blocks. The stored barrier/future CPU window
contains the wire timing window; they are not equal. All four predeclared
gen64 co-primary effects below improve in every pair. Intervals are nominal
two-sided t95, df4, without familywise adjustment or an omnibus significance
claim. Positive gain means RoPE-minus-baseline TPS or baseline-minus-RoPE TPOT.

| Gen64 / clients | primary metric | baseline mean | RoPE mean | paired gain | nominal t95 gain |
|---|---|---:|---:|---:|---|
| 1 | Output tokens/s | 227.783294 | 255.491920 | +27.708625 | [+25.139813,+30.277437] |
| 1 | TPOT ms | 4.373307 | 3.898664 | +0.474644 | [+0.430334,+0.518953] |
| 8 | Output tokens/s | 592.839343 | 650.341016 | +57.501673 | [+51.317801,+63.685545] |
| 8 | TPOT ms | 13.282332 | 12.110283 | +1.172049 | [+1.041118,+1.302980] |

Local mean throughput increases about 12.2% at one client and 9.7% at eight;
TPOT decreases about 10.9% and 8.8%, respectively. These percentages describe this
matched shared-GPU collection, not a fixed isolated speedup or a competitive
serving result. Concurrent requests still have much slower TPOT than a single
request. The original scheduling benefit/cost in finding0023 is a separate
experiment; historical measurements must not be multiplied into one combined
speedup.

| Diagnostic / gen / clients | metric | baseline mean | RoPE mean | nominal t95 gain |
|---|---|---:|---:|---|
| 64 / 1 | TTFT ms | 5.413344 | 4.842059 | [+0.416790,+0.725780] |
| 64 / 1 | Completion ms | 280.931712 | 250.457881 | [+27.594164,+33.353497] |
| 64 / 8 | TTFT ms | 23.655023 | 21.515123 | [+1.058324,+3.221476] |
| 64 / 8 | Completion ms | 860.441929 | 784.462925 | [+66.882149,+85.075858] |
| 1 / 1 | Output tokens/s | 185.330002 | 208.590907 | [+19.419185,+27.102626] |
| 1 / 1 | TTFT/completion ms | 5.379442 | 4.777716 | [+0.515290,+0.688164] |
| 1 / 8 | Output tokens/s | 215.704708 | 245.752898 | [+25.302204,+34.794175] |
| 1 / 8 | TTFT/completion ms | 36.860610 | 32.389853 | [+3.725694,+5.215821] |

Gen1 has no TPOT. Raw stdout also reports pooled descriptive p50/p99; they
are not independent block-level tail estimates or an SLO guarantee. Gen64/
eight-client TTFT p99 is28.978710 ->25.756909 ms in this collection.

Five bootstrap durations average53.686528 s baseline and52.778737 s RoPE,
ranges[52.754304,54.340583] and[52.457346,53.511897]s. The owning ephemeral
compiler cache was already populated by qualification/profile; these are
warm-cache startup observations, not cold compiler-cache costs or startup
performance inference. The capture/bootstrap phase is excluded from active
client timing. Executor-PID CPU totals over 20 cells are0.08 s baseline and0.09 s
RoPE at100Hz accounting; five one-second idle samples total0.01 s and0.00 s.
This coarse scope excludes kernel TX, softirq, client/compiler workers and
total host cost; it does not imply zero total CPU or energy improvement.

[Environment snapshots](../../bench/results/qwen_compiled_rope_environment_20261010.txt)
retain the same other-owner SGLang scheduler with 18,402MiB before, during and
after collection. The active snapshot catches this task's candidate during
startup, not steady-state bank residency or peak memory; no total-memory
reduction is inferred. CUDA profiles above show unchanged PyTorch counters.
Per-cell GPU endpoint clocks/temperature/power/utilization remain in raw.
Alternating pairs and stable residency do not prove co-tenant activity was
constant or establish isolation, physical overlap or integrated energy.

Post-run checks at2026-10-10T20:17:33UTC find no task executor/benchmark process,
module or bpfusion pins. At20:17:45UTC the host has only the preserved SGLang
GPU process and unchanged boot ID. Each driver cleanup completed normally and
eight FREE slots were recorded before detaching. Original Workspace/PVC and
unrelated setup_env.sh remain preserved. This result supports the tested
RoPE-only component hypothesis. Text, continuous batching, optimized serving,
total-host cost and physical-NIC acceptance remain open.


## Independent result review and retained RoPE capture

A fresh read-only result reviewer independently recomputed every request and
cell metric, all five paired intervals, p99/counts/balance/order, oracle agreement,
source identity, common native binary, counters/FREE/cleanup and documentation
numbers. Its verdict is valid, supported separately for all four primary
effects, and supporting engineering value, with no blocker to exact two-line
retention. It checked the default-native strict qualifications and the
unchanged graph/tensor lifetime. It also retained shared-GPU/non-IID,
nominal-interval, warm-cache, CPU-scope and profiling interpretation boundaries.
No separate saved reviewer report is claimed.

The production capture now contains exactly the qualified two lines:
import the existing Qwen2 modeling module and compile apply_rotary_pos_emb with
emulate_precision_casts=True. This non-default compiler option preserves the
fp16 cast boundaries needed by the tested exact oracle; backend and mode
remain defaults. Reduction/projection/attention/KV, scheduler, page/module,
transport and entrypoints are unchanged. No helper, flag, new dependency or
persistent configuration was added. make qwen-graph qwen-graph-control
confirms both canonical symlinks expose the same retained bytes and native
binaries. The [canonical resident regression](../../bench/results/qwen_compiled_rope_canonical_correctness_20261010.txt)
now terminates EXIT0 at the actual build/qwen_graph entrypoint with default
compiler-cache selection: 335 valid requests /20,432 exact tokens /259 rejects,
5,000 empty connections, mixed lengths, reuse, pipeline, half-close, one host
launch /20,432 device graphs /error0 /banks8 /max_active8, zero queue/registration
errors, eight FREE slots, passing all-thread network trace and clean teardown.
The cache directory already existed; this is entrypoint qualification, not a
cold-cache measurement. Production growth is exactly two lines; tests are
unchanged. No full comparison or result review was repeated.


## Reduction-preserving RMSNorm pointwise-tail candidate

The retained RoPE comparison answers its own component question. Its remaining
decode graphs still contain49 RMSNorm instances, each with multiple small
pointwise kernels after its reduction. The previous whole-RMSNorm compiler
failures remain closed; they do not settle this new compilation boundary.
The temporary candidate retains original eager Torch float32 conversion,
pow(2) and mean(-1, keepdim=True), and compiles only epsilon addition, rsqrt,
normalization multiply, cast-back and learned-weight multiply with
emulate_precision_casts=True. The existing eager projection/attention,
precision policy, RoPE, KV/private banks, oracle, scheduler and protocol remain
unchanged. Candidate arithmetic agreement is empirical; passing this probe
does not identify the old whole-RMSNorm failure's cause.

[Exact source and GPU-only qualifier](../../bench/results/qwen_norm_tail_qualification_20261010.txt)
finish EXIT0: all 12 two-bank groups,1,536 exact tokens, ten gen64 prompt-edge
cases and two mixed groups; each group has one host launch/128 model graphs/
error0. No TCP, queue lifecycle or serving benefit follows from this probe.
The [candidate diagnostic profile](../../bench/results/qwen_norm_tail_profile_20261010.txt)
reuses the previous exact RoPE-source profile as its event-count reference.
Five shapes remove196 CUDA events each: prefill5 1005 ->809, prefill64
1077 ->881, decode positions16/64/126 957 ->761. PyTorch allocated/reserved/
held-logits counters remain 1,089,993,728 /1,518,338,048 /57,735,680 bytes.
Sequential CUDA-event times are retained but provide no causal speed claim;
native graph memory and total host/energy remain unmeasured.

## RMSNorm-tail serving comparison plan

This is supporting engineering for the user's throughput/latency goal, no paper
RQ/thesis is invented. The new uncertainty is whether this precision-qualified,
reduction-preserving boundary reduces actual client TPOT and raises throughput
beyond the current RoPE-optimized concurrent path. Positive matched effects
support retaining the nine-line candidate; negative/inconclusive effects leave
production unchanged and direct effort toward request/kernel overlap and
batched computation. Existing RoPE or failed full-norm results cannot answer
this boundary. This is the strongest immediately runnable local computation
comparison; it is not a second architecture contribution or external-serving
win. Actual concurrent tracing/batching remain higher-potential subsequent
work rather than redundant repeats of closed component comparisons.

Baseline is exact production 581999d; candidate differs only by the nine-line
pointwise-tail function and original-reduction wrapper. Both performance
executors use the identical existing eight-bank resident native binary with
the common temporary checked TCP_NODELAY setting. [Frozen adapter/source/commands](../../bench/results/qwen_norm_tail_adapter_20261010.txt)
retain both paths, delta and the reused driver; label/docstring changes are the
only changes from the completed RoPE protocol. Default resident and serial-host
native regressions are separate correctness controls, not performance baselines.
Both strict gen64/prompt-edge regressions and a real preflight must terminate
successfully before full collection; they retain every oracle/lifecycle/trace
assertion and request/shutdown timeout.

Reuse the fixed RoPE matrix and metric definitions above: four prompts,
gen1/64, clients1/8, five alternating paired process blocks, matching rotating
cell order,64 requests/cell, same warmups and idle sampling. Complete means all
40 cells/2,560 requests/83,200 exact tokens/ten normal exits, with each process
264 publications/8,580 model graphs/one host launch/error0/banks8/max_active8,
zero queue/registration errors, eight FREE slots and cleanup. The compiler
cache is the existing owning ephemeral cache, populated by qualification/profile;
record startup separately and make no cold-cache claim.

Assess all four gen64 TPOT/TPS primary effects separately with all five paired
blocks and nominal two-sided t95, df4. Positive gain means baseline-minus-tail
TPOT or tail-minus-baseline TPS. Mixed effects remain mixed; intervals crossing
zero are inconclusive, not equivalence. Report TTFT, completion, gen1, pooled
descriptive tails, startup and executor-PID CPU as diagnostics. TPS uses
first-send/last-arrival, not the broader barrier/future CPU window; metric
definitions remain the linked NVIDIA convention above. No weighted score,
omnibus significance, post-hoc cell selection or exclusion is added.

Preserve the SGLang co-tenant and record coarse host residency/boot and
per-cell GPU endpoints before/after. These do not prove constant activity,
IID/isolation, physical kernel overlap, total memory/energy or total-host cost.
No physical-NIC, open-load capacity, optimized-serving or fixed isolated-speedup
claim follows. A fresh read-only plan reviewer and one fresh terminal result
reviewer assess this local retention decision; no separate saved report is
claimed. At plan freeze, production remained581999d pending the complete comparison and review.


Fresh read-only plan review admitted the frozen RMSNorm-tail comparison,
conditional on both default strict qualifications and real preflight passing.
It independently verified the nine-line boundary, original installed HF
reduction, exact source/common native binary, independent oracle, unchanged
Graph/tensor ownership, metrics/counts/protocol and minimal code growth.
It found no scientific/executability blocker. All four primary effects,
shared-GPU/non-IID and nominal intervals, warm-cache startup and complete
counter/FREE/cleanup verification remain required. The inherited driver
comment about independent blocks is not evidence of IID. No separate saved
reviewer report is claimed.


Both default-native candidate strict regressions now finish EXIT0:
[resident](../../bench/results/qwen_norm_tail_resident_correctness_20261010.txt)
and [serial host control](../../bench/results/qwen_norm_tail_host_correctness_20261010.txt),
each 335 valid requests /20,432 exact tokens /259 rejects,5,000 empty
connections, eight-client invalid-to-valid reuse, mixed decode lengths,
pipeline/half-close/ring wrap, zero queue/registration errors, eight FREE slots
and passing all-thread network trace/cleanup. Resident records one host
launch/20,432 device graphs/error0/banks8/max_active8; host records 594
launches/the same model-graph count/error0/banks1/max_active1. No oracle,
timeout or lifecycle assertion was weakened. Real preflight completion is recorded below;
At qualification, production remained the retained RoPE-only source581999d.


Real preflight now finishes EXIT0:
[raw](../../bench/results/qwen_norm_tail_preflight_20261010.jsonl) /
[stdout](../../bench/results/qwen_norm_tail_preflight_20261010.stdout.txt),
15 records/eight exact gen1 requests/two normal exits. Each process has12
publications/264 model graphs/one host launch/error0/banks8/max_active1,
zero queue/registration errors and eight FREE slots; module/pins are absent
after cleanup. Compiler cache is already populated. This is executability
evidence, not a serving result. Both strict qualifiers and real preflight
close every plan-review condition; the frozen full matrix can now run.

## Complete RMSNorm-tail TCP comparison

The full command in the plan ran from2026-10-10T20:34:03UTC to its normal
EXIT0 on the unchanged frozen path. Collection metadata identifies
b5b6774d4008d815d22a05b2d13377ba34b18759; baseline Python is the retained
RoPE source581999d and candidate adds only the nine lines above. Both engines
use the same eight-bank native library with the documented common temporary
TCP_NODELAY transport. The production native library and module are unchanged.

[Full raw](../../bench/results/qwen_norm_tail_20261010.jsonl),
[command/stdout](../../bench/results/qwen_norm_tail_20261010.stdout.txt),
[recomputation and cleanup](../../bench/results/qwen_norm_tail_analysis_20261010.txt)
and [host observations](../../bench/results/qwen_norm_tail_environment_20261010.txt)
retain all five pairs:40 cells,2,560 measured requests,83,200 exact output
tokens and10 normal executor exits. Every cell has64 requests with16 of each
of the four frozen prompts. All request tokens match the independent HF eager
oracle. Every process records264 publications,8,580 model-graph launches,
one host launch,graph_error=0,banks=8,max_active=8,zero queue/parser/sockhash
errors and eight final FREE slots. Warmups remain excluded.

Positive gain means candidate-minus-baseline TPS or baseline-minus-candidate
TPOT. The four predeclared gen64 primary effects are each positive in all
five pairs. These are nominal two-sided paired t95 intervals with df4;
no familywise,omnibus,isolated-GPU or optimized-external-serving claim follows.

| gen64 clients | primary metric | RoPE baseline | RMSNorm tail | mean gain | nominal t95 gain |
|---:|---|---:|---:|---:|---|
| 1 | Output tokens/s | 254.819208 | 285.217876 | +30.398668 | [+27.412444,+33.384891] |
| 1 | TPOT ms | 3.909868 | 3.494420 | +0.415448 | [+0.379685,+0.451211] |
| 8 | Output tokens/s | 647.939413 | 674.521571 | +26.582158 | [+18.812551,+34.351764] |
| 8 | TPOT ms | 12.153415 | 11.677643 | +0.475773 | [+0.329880,+0.621665] |

Within this comparison,mean TPS increases about11.9% and4.1%, while TPOT falls
about10.6% and3.9% at clients1 and8 respectively. Concurrent request TPOT
remains substantially slower than single-client TPOT. Earlier findings have
their own environments; historical gains must not be multiplied into a
combined isolated speedup.

The remaining metrics are diagnostics, not additional co-primary effects.

| gen / clients | metric | baseline mean | tail mean | nominal t95 gain |
|---|---|---:|---:|---|
| 64 / 1 | TTFT ms | 4.799451 | 4.218178 | [+0.443641,+0.718904] |
| 64 / 1 | Completion ms | 251.121117 | 224.366636 | [+24.395838,+29.113124] |
| 64 / 8 | TTFT ms | 21.743985 | 20.737825 | [+0.597552,+1.414768] |
| 64 / 8 | Completion ms | 787.409155 | 756.429307 | [+22.071489,+39.888207] |
| 1 / 1 | Output tokens/s | 211.625159 | 242.681430 | [+25.239645,+36.872898] |
| 1 / 1 | TTFT/completion ms | 4.713365 | 4.106300 | [+0.485208,+0.728921] |
| 1 / 8 | Output tokens/s | 249.098719 | 274.517304 | [+16.803149,+34.034021] |
| 1 / 8 | TTFT/completion ms | 31.943516 | 28.973013 | [+2.014279,+3.926728] |

Gen1 has no TPOT. Pooled descriptive gen64/eight-client TTFT p99 is
25.967602ms baseline and24.954841ms tail; this is not a block-level tail
estimate or general SLO guarantee. Client tokens completed in the same recv
share an arrival timestamp.

Five bootstrap durations average53.328557s baseline
(range52.253011..56.523941) and53.090387s tail
(range52.150493..53.875994). The compiler cache was already populated by
qualification/profiling; this is warm-cache startup, not a cold compilation
cost comparison. Executor PID CPU totals over20 cells per mode are0.08s
baseline and0.06s tail. Five separate one-second idle samples per mode record
0.00s CPU at100Hz/0.01s accounting resolution. These coarse samples do not
establish zero CPU work,precise savings,total host/kernel cost or energy.

The same SGLang scheduler hostPID732929 is observed before,during andafter
collection with18,404MiB GPU memory. Endpoint clocks,temperature,power and
utilization cannot establish stationarity or within-window co-tenant activity.
The during snapshot follows the first baseline's exit and catches the next
candidate at startup; its2,246MiB is not steady bank residency,peak memory or
an allocation delta. The one-bank profile above reports unchanged PyTorch
allocated/reserved/held-logits bytes, not an eight-bank serving-memory comparison.

Post-run Workspace checks at20:47:53UTC find the module and bpfusion pins absent
and this experiment's parent/executor processes gone. Host observations at
20:48:06UTC find only the existing SGLang GPU process and the unchanged boot
ID. Every driver cleanup call completed, with FREE slots recorded before
detaching. No co-tenant was stopped or changed.


## Independent RMSNorm result review and retained capture

A fresh read-only result reviewer independently recomputed all four primary
effects and diagnostics from the full raw, checked the frozen sources against
their commits, inspected the installed HF reduction and the pre-patch oracle,
and audited qualification,profile engagement,source/native identity,ordering,
lifetimes and cleanup. The review finds this run valid and supporting for the
tested component/workload comparison, with no publication blocker. It does not
complete the overall service goal or explain the older whole-RMSNorm failures.

The exact nine-line candidate is now integrated into executor/qwen_graph.py.
No test,CUDA/native,module,queue ABI,transport,dependency,mode flag or persistent
helper changes accompany it. make qwen-graph qwen-graph-control validates the
existing canonical entrypoints. The
[canonical regression](../../bench/results/qwen_norm_tail_canonical_correctness_20261010.txt)
uses default compiler-cache selection and unchanged default native networking,
with HF_HOME selecting the existing model cache. The default compiler-cache
directory already existed; no cold-cache cost is claimed. It terminates EXIT0
with335 valid requests,20,432 exact tokens,259 rejects and5,000 empty
connections,including prompt edges,mixed lengths,wrap/reuse,rejection recovery,
pipeline and half-close. It records one host launch,error0,banks8,max_active8,
no accept/receive/send syscalls on all traced executor threads after ready,
eight final FREE slots and module/hook/pin cleanup. Both default-native modes
were already qualified on the byte-identical candidate above; this final run
qualifies the retained canonical path and cache selection.

The completed comparison supports retaining this reduction-preserving boundary
for this shared-GPU workload. The next useful uncertainty is actual concurrent
model-kernel execution and the costs limiting batching/individual decode;
profile event reduction and max_active8 alone do not answer it.

## Next computation candidate: merged MLP gate/up projection

The complete device-graph diagnostic in finding0023 observes cross-graph
model-kernel activity overlap and attributes most summed kernel duration
to GEMV name groups. Those names do not map individual shapes and do not
prove memory saturation. The largest directly runnable next computation
change is to combine the gate/up MLP projections,which read the same
input and have equal output sizes,into one ordinary linear operation.
Concatenate their weight rows once after model loading,split the projected
output,then retain the original SiLU,multiply and down projection.
The mathematical expression is unchanged,but the larger linear shape can
change numerical reduction/algorithm selection; exact qualification is
necessary. No new precision setting or relaxed oracle is permitted.

No paper RQ/thesis exists. This is a supporting engineering comparison for
the user's throughput and latency goal: can this merged projection preserve
the independent eager-fp16 oracle and lower client TPOT/increase total output
TPS versus the current retained computation? It tests a new linear-shape
change,not a repeat of the closed RoPE/RMSNorm or dispatch comparisons.
True tensor batching previously failed exact qualification,and there is no
validated repair for it yet. This candidate changes a dominant operation
class with nine temporary source lines,without a native/module/queue change
or a new persistent mode. If it diverges,retain the complete qualification
failure and discard the candidate without collecting a misleading speed
comparison. If it qualifies but performance is inconclusive or worsens,
report all cells/costs and retain current computation.

Baseline is current production atc17c440 (same computation asfc58faa).
Candidate is the frozen source at
/tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_mlp/qwen_graph:
the baseline plus nine bootstrap/forward lines. The gate projection receives
the concatenated gate/up weights and doubled output metadata; the original
up module is removed after concatenation,so model weights are not retained
twice. The capture/graph-bank/KV/math outside that linear shape,weights file,
transport,strict oracle and lifecycle stay unchanged. The source/model
objects remain owned through native shutdown. Actual HF Linear continues
to choose its default algorithm; no compiler/precision/tuning fallback is
introduced. This is a current-practice component comparator,not an external
optimized-serving baseline.

Qualification folds into this comparison's preflight. First run the existing
native graph-bank probe using qualify_merged_mlp.py with two private banks:
all ten gen64 fixed correctness/prompt-edge cases,one mixed France/greeting
group and one mixed-prompt-length group,12 groups/1,536 tokens. It retains
every failing row; marker completion is not a correctness pass. Independent
oracle_cases(64,True) runs before transformed model capture. The probe is
GPU computation qualification only,not a serving result. If all pass,run
the documented tests/resident_qwen.py --gen64 --prefill-edges against the
default-native candidate and the existing serial host-control entrypoint:
335 valid requests/20,432 exact tokens/259 rejects per mode,empty connections,
mixed lengths,wrap/reuse,pipeline,half-close,all-thread network trace and
native counters/cleanup. No correctness or request/shutdown timeout changes.
Both modes use the unchanged built native libraries.

After qualification,real TCP comparison preflight and full run reuse the
completed serving driver's request/connect/cell/oracle protocol. Frozen
merged_perf_baseline and merged_perf_candidate Python sources use the same
existing bank-comparison native library with common temporary listener
TCP_NODELAY. Both baseline and candidate keep the retained RoPE/RMSNorm
fusion. This transport setting is test-local,not a production change.
Real preflight is gen1/client1/four requests per engine,with inherited eight
warmups; readiness follows ready or process exit. Full matrix is gen1/gen64,
clients1/8,five paired process blocks with alternating engine order and
rotating cell order,64 requests per cell,four prompts16 times each,40 cells,
2,560 requests,83,200 measured tokens,ten independent starts/shutdowns.
Keep every block and failed run; no post-hoc exclusions. Bootstrap and
one-second idle samples are recorded separately,not in steady-state timing.

Primary metrics use the existing
[NVIDIA serving definitions](https://docs.nvidia.com/nim/benchmarking/llm/metrics):
gen64 client TPOT=(last-first)/63 and total output TPS over first-send/
last-arrival at clients1 and8. Report all five paired block effects and
nominal two-sided t95 intervals,df4,baseline-minus-candidate for latency and
candidate-minus-baseline for TPS. Judge the four primary effects separately;
a mixed outcome stays mixed. TTFT,completion,gen1,pooled descriptive tails,
bootstrap,executor-only CPU and available memory/environment observations
are costs/diagnostics,not substitutes for correctness or silent vetoes.
This tests a local component/workload effect,not isolated speed,energy,
whole-host CPU,open-load capacity,physical NIC or external serving superiority.

Commands in the original Workspace use the inherited default cache:

    HF_HOME=/workspaces/.cache/huggingface python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_merged_mlp.py 2
    HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_mlp/qwen_graph
    HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_mlp_host/qwen_graph_host_launch
    HF_HOME=/workspaces/.cache/huggingface python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_mlp_bench.py --gen 64 --output bench/results/qwen_merged_mlp_20261010.jsonl --resident-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_perf_baseline/qwen_graph --host-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/merged_perf_candidate/qwen_graph

Add --preflight with its own output to the last command for executability.
Exact sources/commands and terminal raw are retained in cited artifacts.

Shared RTX5090 residency and boot identity are recorded through the existing
host diagnosis path. Preserve the other owner's SGLang workload. This
candidate and its native probes run only in the original Workspace,never
in a new deployment. Request/shutdown sixty-second handling stays fixed;
normal cleanup removes this experiment's module/hooks/pins and restores
FREE slots. At plan admission,source preparation was compile-only. Fresh plan and
terminal result reviews apply; no production retention occurs before
correctness and complete result review.

Fresh independent read-only plan review admits this computation experiment
with no scientific or executability blocker. It inspects source/native
geometry and owner lifetimes,independent oracle,all12 qualification rows,
the distinct eight-bank strict tests,both identical performance libraries,
five-pair metrics/signs and shared-GPU limits. The larger linear shape
remains numerically unproven. GPU probe EXIT0 alone is matrix completion;
every row must match before proceeding. Production growth would be nine
lines if correctness and the complete performance comparison support it;
no production source change is retained at plan admission.

The two-bank GPU qualification completes EXIT0 with all12 groups passing,
1,536 exact tokens and no mismatch. Every native probe reports one host
launch,128 model graphs and error0. The
[complete qualification raw](../../bench/results/qwen_merged_mlp_gpu_qualification_20261010.txt)
retains the independent oracle,all actual tokens and terminal status;
[exact adapter/qualifier/driver](../../bench/results/qwen_merged_mlp_adapter_20261010.txt)
is frozen before execution. This qualifies the linear-shape computation
in the recorded groups,not eight-bank TCP serving or a speed improvement.

Both default-native real TCP strict modes now terminate EXIT0:
[resident raw](../../bench/results/qwen_merged_mlp_resident_correctness_20261010.txt)
and [serial host-control raw](../../bench/results/qwen_merged_mlp_host_correctness_20261010.txt).
Each passes335 valid requests/20,432 exact tokens/259 rejects,5,000 empty
connections,mixed lengths/immediate reuse,ring wrap,pipeline/half-close,
all-thread network tracing and cleanup. Resident counters are one host
launch,eight banks/max_active8; host control is594 launches,one bank/
max_active1. Both record20,432 model graphs,error0,594 publications,zero
admission/registration errors and eight FREE slots. Builds use the documented
make qwen-graph qwen-graph-control entrypoint with no native rebuild needed.
This qualifies the integrated candidate,not a performance improvement.

The real paired TCP preflight terminates EXIT0:
[JSONL](../../bench/results/qwen_merged_mlp_preflight_20261010.jsonl)
and [stdout](../../bench/results/qwen_merged_mlp_preflight_20261010.stdout.txt),
15 records,eight exact measured gen1 requests,two normal executor shutdowns.
Each engine records12 publications,264 model graphs,one host service launch,
error0,eight banks/max_active1,eight FREE slots and cleanup. This establishes
the actual comparison path only; its four-request timings are not a speed
result. The admitted full40-cell comparison was then started on unchanged frozen
sources,oracle,transport and request/shutdown handling.

## Complete merged-projection comparison: mixed result

The [full JSONL](../../bench/results/qwen_merged_mlp_20261010.jsonl)
and [stdout](../../bench/results/qwen_merged_mlp_20261010.stdout.txt)
terminate EXIT0 with2,621 records,all40 cells,2,560 measured requests,
83,200 exact measured tokens and ten normal executor shutdowns.
Every process records264 publications,8,580 model graphs,one host service
launch,error0,eight banks/max_active8,zero admission/registration errors
and eight final FREE slots. Every paired block is retained,with no failed
full cell or post-hoc exclusions. Qualification and preflight dependencies
above remain separately labeled.

The [supplementary analysis](../../bench/results/qwen_merged_mlp_analysis_20261010.txt)
retains the exact CPU analyzer and independently recomputes every token,
arrival order,TTFT,completion,TPOT,cell first-send/last-arrival TPS,prompt/index
balance,process order,counters and the nominal paired intervals below.
Values are means of five block metrics; TPS is not a pooled rate.
Positive gain means lower latency or higher TPS. Nominal t95,df4 intervals
are conditional on this shared-GPU collection,with no familywise adjustment.

| gen64 primary / clients | current mean | merged mean | paired gain | nominal t95 gain |
|---|---:|---:|---:|---|
| TPOT ms /1 | 3.476161 | 3.336241 | +0.139921 | [+0.127406,+0.152435] |
| Output TPS /1 | 286.710208 | 298.455231 | +11.745023 | [+10.654985,+12.835061] |
| TPOT ms /8 | 11.435397 | 11.514194 | -0.078797 | [-0.173426,+0.015832] |
| Output TPS /8 | 688.738071 | 683.955194 | -4.782877 | [-10.450662,+0.884907] |

Single-client gen64 TPOT falls about4.03% and TPS rises about4.10%,with
all five paired effects positive and both intervals excluding zero.
At eight clients,both primary mean effects are worse in all five pairs;
their nominal intervals still cross zero. This is no detected eight-client
benefit,not proof of equivalence or of a fixed regression magnitude.
The mixed result does not support a general throughput/latency improvement.

| paired block | clients1 TPOT gain ms | clients1 TPS gain | clients8 TPOT gain ms | clients8 TPS gain |
|---:|---:|---:|---:|---:|
| 0 | +0.130911 | +10.848507 | -0.001416 | -0.492749 |
| 1 | +0.148381 | +12.559280 | -0.029260 | -1.411271 |
| 2 | +0.143264 | +11.932351 | -0.108662 | -6.667586 |
| 3 | +0.127604 | +10.804974 | -0.059539 | -3.588945 |
| 4 | +0.149443 | +12.580002 | -0.195108 | -11.753836 |

Single-client gen64 completion falls223.184256 ->214.399029ms,
gain8.785227ms [7.966568,9.603886]. TTFT4.186085 ->4.215851ms is
inconclusive,with gain interval[-0.180240,+0.120708].
Eight-client completion740.863849 ->746.066916ms and TTFT20.433838
->20.672686ms are also inconclusive; their gain intervals are
[-11.368701,+0.962566] and[-0.676527,+0.198831].
Gen1/client1 TPS246.806370 ->244.967377 and TTFT4.040695 ->4.068046ms
are inconclusive. Gen1/client8 TPS272.426749 ->279.004142 and
TTFT29.118105 ->28.526519ms improve locally,with gain intervals
[4.505897,8.648887] and[0.289881,0.893289]. These one-token diagnostics
do not replace the declared gen64 primary effects.

Five warm/default-cache bootstrap durations average52.774309s current
(range52.148424..53.707051) and53.378369s merged
(range52.404482..55.576147). No startup benefit is established.
Across twenty active cells per mode,executor CPU totals0.07s and0.08s,
at100Hz/0.01s resolution; the five one-second idle samples are0.00s each.
These coarse executor-only values establish neither total-host cost nor
zero CPU/energy. No matched peak-memory delta was measured.

[Host snapshots](../../bench/results/qwen_merged_mlp_environment_20261010.txt)
before qualification,before full collection and after collection identify
the same boot and untouched SGLang owner process with18,404MiB residency.
All per-cell endpoints remain in JSONL. Residency/endpoints do not establish
constant within-window co-tenant activity or isolate the component effect.
Post-run inspection finds the module/pins absent; no other owner was changed.
The exact candidate is preserved in raw,but current production computation
remains unchanged after the single fresh terminal result review.
This is a supporting linear-shape/workload boundary,not a new complete
serving win or evidence of particular GEMV bandwidth/occupancy causes.

### Fresh terminal result review and decision

The independent result reviewer read the admitted plan, candidate/qualifier/
driver source, all raw records, the analyzer, native transport evidence and
environment snapshots without rerunning GPU work or editing artifacts. It
independently recomputed every primary metric, paired effect and interval,
and checked all qualification tokens and lifecycle counters.

Its verdict is **run valid; overall hypothesis inconclusive, single-client
sub-effect supported; supporting research value; mechanism/workload boundary**.
The complete matrix has no failure, omitted or excluded cell. Correctness,
source identity, matched transport, alternating process order and cleanup
all pass. The five gen64/eight-client point estimates are negative, while
their intervals cross zero: neither a fixed regression nor equivalence is
established. The positive gen1/eight-client secondary outcomes do not replace
the gen64 primary outcomes.

The decision is to **retain the current production computation and close the
merged-projection candidate without installing it as the default**. The
single-client improvement remains a valid local result; an unconditional
eight-bank optimization is unsupported. No new serving mode, persistent
helper or precision change is added. Production and test code changes for this
experiment are both zero; the nine-line temporary candidate and complete raw
comparison remain reproducible evidence. Do not repeat this unchanged matrix.
The next computation investigation should distinguish a different mechanism
that can improve the concurrent path.

## Next attention candidate: grouped-query matmuls without repeated KV

The completed device trace shows actual graph overlap. The operation diagnostic
associates decode GEMV names with several families and records explicit
[1,2,7,context,64] KV-head replication copies. Closed merged MLP results do
not support an eight-bank default improvement. A different attention mechanism
can eliminate those copies while retaining the original explicit score,
mask,fp32 softmax,fp16 output and dropout expressions. This tests whether
avoiding replicated KV materialization improves the actual concurrent TCP
path. Positive and negative outcomes change implementation retention; the
earlier traces and rejected SDPA cannot answer this different computation
shape. This is supporting engineering evidence within the user's performance
goal,not a paper RQ,headline serving comparison or repeated dispatch experiment.

The fourteen-line temporary candidate groups the seven query heads associated
with each KV head into rows. Instead of replicating each key/value head seven
times,it computes score bmm with batch*KVheads and rows=groups*query_length,
restores the original [batch,query_heads,length,context] score shape,and keeps
the installed HF scaling,mask addition,float32 softmax,cast and dropout.
The weighted-value bmm uses the corresponding original KV value head and
restores the original returned output/attention shapes. No precision,
attention-mask semantics,RoPE,RMSNorm,KV ownership,queue/module/native
dispatcher or transport change is proposed. Input reshape may itself copy,
particularly for prefill; larger bmm row shapes may select different numerical
algorithms. Algebraic equivalence is not an exact-token qualification.

Scope: source preparation,tests and measurement only in the existing
bpf-gpu-server Workspace,current main. Production0/test0 at admission. Baseline is
the byte-exact current qwen_graph.py (also fc58faa). The temporary candidate
is /tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa/qwen_graph; its host
qualification entry uses canonical build/qwen_graph_host_launch.so.
Both resident performance entries grouped_gqa_perf_baseline/qwen_graph and
grouped_gqa_perf_candidate/qwen_graph use the same previously checked
test-local TCP_NODELAY native .so. Original canonical default transport is
used for both strict qualifiers. Nothing is installed as a new service,
mode,helper,configuration or model.

First run the reused independent eager-fp16 oracle qualification:
HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_grouped_gqa.py 2.
It constructs oracle_cases(64,True) before candidate patching and uses the
existing real device-launched bank probe. Preserve all12 groups,1536tokens
and mismatches. A failed row closes this candidate without speed collection,
oracle relaxation or repeating an unchanged qualification. If all pass,
run tests/resident_qwen.py for the candidate resident and host entries:
335 valid requests,20432 exact tokens,259 rejects each,with mixed lengths,
reuse,wrap,pipeline,half-close,5000 empty connections,network trace and cleanup.
The old host variant is a serial correctness control,not the performance
baseline. Use existing make qwen-graph qwen-graph-control if builds are needed.

The actual performance driver is
/tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa_bench.py,adapted only by
mode/banner names from the completed merged comparison. Run --gen64 --preflight with
--resident-executor pointing to the frozen baseline and --host-executor to
the frozen candidate,output bench/results/qwen_grouped_gqa_preflight_20261010.jsonl.
Full command omits --preflight and uses
bench/results/qwen_grouped_gqa_20261010.jsonl. Both engines are resident.
The full matrix remains five paired process blocks with alternating engine
order,matching rotated gen1/64 x clients1/8 cells,64 measured requests/cell,
four prompts16 times/cell:40cells2560requests83200tokens. Each process has
eight excluded warmups,264publications,8580modelgraphs,one service launch,
zero errors and final FREE slots. Bootstrap and separate idle samples are
outside measurement; existing request/shutdown60s and readiness-or-exit
behavior remain unchanged. Preserve failures and all cells.

Primary outcomes remain gen64 TPS and TPOT at clients1/8,using the existing
[NVIDIA metric definitions](https://docs.nvidia.com/nim/benchmarking/llm/metrics)
and five paired block means/nominal df4 t95 intervals. Positive gain is
candidate-minus-baseline TPS and baseline-minus-candidate TPOT. Recompute
timestamps/cell windows/oracle/counters with the saved analyzer; do not treat
requests as independent process repetitions. Four effects remain separate
nominal comparisons without omnibus superiority or equivalence claims.
Retain a qualified candidate only if the concurrent gen64 throughput and TPOT
both support improvement and no primary effect supports a regression; report
any inconclusive single-client effect separately. Mixed/inconclusive
concurrent results retain the existing default. Secondary TTFT,completion,
gen1,bootstrap,memory and executor-only CPU describe tradeoffs and cannot
replace the primary outcomes. A profiler event reduction is a mechanism
control,not a serving win.

Record exact source/helper provenance,command,environment and all raw outputs
in existing finding0021 and cited ordinary bench/results artifacts. Matching
temporary transport,shared RTX5090/co-tenant,same short prompts,closed-loop
load,greedy oracle and coarse executor CPU bound the claim; no isolated
bandwidth,physical NIC,open-load capacity,total-host cost,energy or optimized
serving advantage is established. Preserve other owners and verify process/
module/pin cleanup. Fresh plan and terminal result reviews apply.


Concrete performance command (run preflight first by appending --preflight
and using the preflight output filename; no change to the64-token oracle):

    HF_HOME=/workspaces/.cache/huggingface python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa_bench.py --gen 64 --output bench/results/qwen_grouped_gqa_20261010.jsonl --resident-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa_perf_baseline/qwen_graph --host-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa_perf_candidate/qwen_graph

Strict command for each candidate entry:

    HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/grouped_gqa/qwen_graph

For host qualification,substitute grouped_gqa_host/qwen_graph_host_launch
only. Exact candidate,qualifier,driver and analyzer are frozen in
bench/results/qwen_grouped_gqa_adapter_20261010.txt. The recorded collection
base is4da46fa9f76ebfd160b22aff099d25574a5ad20f,whose production is still
byte-exactfc58faa. Helper functions and independent oracle are inherited
unchanged from the already-published merged comparison.

Fresh read-only plan review admits the grouped-query candidate with no
scientific or executability blocker. It verifies the installed HF eager
global lookup actually invokes the patch,14/2/7 head ordering,all return
shapes,valid KV prefix,unchanged score/softmax/cast semantics,independent
oracle ordering and exact64-token commands. The larger bmm shape still
requires empirical qualification. Both performance natives and all inherited
client/CPU/window helpers match;14temporary lines,production0/test0.

### Grouped-query qualification result: rejected numerical path

The real GPU-only device-launched qualification exits0 and completes all12
groups/1536 generated tokens,but **five groups fail the unchanged exact-token
oracle**. Exit0 and matrix completion are not a pass. Groups0,2,4,mixed5
and mixed_lengths fail; seven groups pass. The independent eager batch1
oracle was computed before candidate patching.

- Case0 first differs at zero-based output13: actual3283,expected304.
- Case2 first differs at output62: actual11,expected323.
- Case4 first differs at output31: actual7010,expected36277.
- Mixed groups preserve those failures,including lane0/case0 in mixed_lengths.

[All actual tokens,mismatch rows,oracle and native counters](../../bench/results/qwen_grouped_gqa_gpu_qualification_20261010.txt)
and [frozen sources](../../bench/results/qwen_grouped_gqa_adapter_20261010.txt)
remain intact. Each group records one host launch,128 model graph executions
and error0; transport was not involved. Mathematical head grouping does not
preserve this greedy numerical reference under the actual larger bmm shapes.
The result does not identify which operation/rounding path caused divergence
or prove every mathematically equivalent grouped implementation must fail.

The decision is to close this candidate without default-native strict TCP
tests,preflight or speed collection,as declared at admission. Production and
tests remain unchanged; no precision/oracle adjustment,fallback or new mode
is added. No serving speed is claimed. [Host observations](../../bench/results/qwen_grouped_gqa_environment_20261010.txt)
show the same boot and preserved SGLang owner. Owned qualification processes,
module and pins are absent after exit. Fresh terminal result review is complete.


The fresh independent read-only terminal review recomputes all tokens and
first mismatches:five failed groups,nine failed lanes,seven passing groups,
1536 generated tokens. Every native marker and source identity agrees.
Verdict: valid completed qualification;exact-reference premise contradicted
for this candidate;dependency-only evidence,mechanism/workload boundary.
The speed hypothesis is untested. Skipping strict TCP and performance is the
predeclared branch,not an incomplete admitted speed run. It confirms the
HF patch is actually invoked,independent oracle precedes patching,and no
owned work/module/pins remain. Publish the failure and keep the default.

## Next norm candidate: fuse conversion and square,retain eager mean

Whole-norm compilation failed the fixed oracle; retained norm-tail fusion
succeeded by leaving its reduction eager. The completed shape diagnosis
records49 hidden-state conversion kernels and49 pow kernels per model graph.
A distinct pointwise boundary can combine those first two operations while
keeping exactly the original mean reduction and the retained norm tail.
This tests whether fewer conversion/square device kernels improve concurrent
TCP throughput/TPOT without changing greedy outputs. It adds evidence beyond
the closed tail fusion,whole-norm failures and grouped-attention rejection.
Positive results support retention; failure or inconclusive concurrent
effects preserve current computation. This is supporting engineering work
within the unchanged user performance goal,not a new paper RQ or a complete
serving claim. It is a runnable computation improvement with numerical risks
bounded by the independent oracle; another unqualified batching/matmul change
does not offer stronger currently validated evidence.

Scope: original bpf-gpu-server Workspace/main,no production/native/module/queue/
oracle/transport change at admission. Current basea98a316c7b992e5426fcea6584f57c2510795220
has production byte-exactfc58faa. Temporary norm_prepare/qwen_graph adds four
lines net: a default-backend compile function returns both original float32
hidden_states and hidden_states.pow(2),using the same emulate_precision_casts
option already required by retained pointwise fusions. norm_forward consumes
those tensors and executes squared.mean(-1,keepdim=True) eagerly with the
original dtype/shape. norm_finish,epsilon,weight/cast and all model operations
otherwise remain unchanged. The additional output is needed by the unchanged
norm tail,not a new retained buffer/interface. Compiled arithmetic and strides
still require real qualification; no equivalence is assumed.

Exact source/qualifier/driver/analyzer/mechanism profiler are frozen in
bench/results/qwen_norm_prepare_adapter_20261010.txt. Preparation adds
production0/test0. Temporary root is
/tmp/bpfusion-qwen-pointwise-5mxeys8g. Candidate host entry
norm_prepare_host/qwen_graph_host_launch uses the canonical native control.
Performance entries norm_prepare_perf_baseline/qwen_graph and
norm_prepare_perf_candidate/qwen_graph use the same existing checked test-local
TCP_NODELAY native library. Client/oracle/CPU/window helpers are unchanged
from the already-reviewed comparisons. No new mode,precision,compiler tuning,
persistent helper or infrastructure setting is introduced.

Reuse the preceding candidate's full fixed protocol and metric definitions:
12GPU qualification groups/two banks/1536tokens first; then both canonical
strict modes335valid/20432exact/259reject with empty/reuse/mixed/wrap/pipeline/
half-close/network-trace/cleanup; actualTCPgen1 preflight; and the complete
five-pair40cell2560request83200token gen1/64 x clients1/8 comparison.
If any qualification row fails,close the candidate without speed collection.
Never relax oracle or treat EXIT0/matrix completion as qualification pass.
Five alternating process pairs and matching rotated cells,all warmup/idle/
bootstrap exclusions,request/shutdown60s handling and ready-or-exit behavior
remain unchanged. Preserve negative/inconclusive/failure rows and all costs.

Commands from repository root,HF_HOME=/workspaces/.cache/huggingface:

    python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_norm_prepare.py 2
    python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/norm_prepare/qwen_graph
    python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/norm_prepare_host/qwen_graph_host_launch
    python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/norm_prepare_bench.py --gen 64 --preflight --output bench/results/qwen_norm_prepare_preflight_20261010.jsonl --resident-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/norm_prepare_perf_baseline/qwen_graph --host-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/norm_prepare_perf_candidate/qwen_graph

For full comparison omit --preflight and use qwen_norm_prepare_20261010.jsonl.
After qualification,the inherited GPU-only profile_norm_prepare.py optionally
verifies event-count mechanism at the same five graph indices as the retained
norm diagnostic. It captures one bank and replays graphs on the host; compare
event counts with already-retained qwen_norm_tail_profile_20261010.txt,not
sequential latency/allocations as a serving benefit. Profile is supporting
mechanism evidence folded into this experiment,not a separate run contribution.

Primary gen64 TPS/TPOT effects remain separate nominal paired df4 t95 results
using five block means,positive gain=candidate-minus-baseline TPS and baseline-
minus-candidate TPOT. Retain only if both concurrent primary outcomes support
improvement and no primary effect supports regression; report single-client
inconclusive effects separately. Secondarygen1,TTFT,completion,CPU,memory and
bootstrap cannot replace that decision. Independent oracle and all actual
timestamps/counters are checked. Warm compiler cache,common temporary transport,
shared GPU/other owner,short prompts,closed loop and coarse executor CPU are
limits; no total-host CPU,energy,isolated bandwidth,NIC,optimized-serving or
whole-RQ claim. Fresh plan and result reviews apply. Preserve other owners,
inspect boot/residency and owned process/module/pin cleanup,and publish only
the completed branch and unchanged numerical standard.

Fresh read-only plan review admits the conversion/square experiment. It
confirms the compile boundary excludes mean,norm input dtype/epsilon/tail
are unchanged,all frozen sources and common natives match,and the
independent oracle runs before patching. The multi-output compiled function
still requires actual qualification;kernel reduction and speed remain
unproven. No additional baseline,framework or configuration is needed.

The real two-bank GPU qualification completes all12 groups/1536 exact tokens,
with no mismatch. Each group records one host launch,128 actual model graphs
and error0. The independent oracle precedes candidate installation; full
actual token rows and prompt edges/mixed groups are preserved in
bench/results/qwen_norm_prepare_gpu_qualification_20261010.txt.
This establishes this numerical prerequisite,not TCP correctness or speed.
The admitted default-native resident strict regression completes EXIT0.


The canonical resident strict test passes335 valid TCP/HTTP requests,
20432 exact generated tokens and259 rejects. It includes5000 empty connections,
mixed gen1/64,immediate reuse,ring wrap,pipeline and half-close. Native counters
record one host launch,20432 model graphs,error0,banks8/max_active8,
594publications,zero queue/registration errors and eight FREE slots.
The all-thread trace has no executor accept/receive/send syscall after ready;
normal module/hook/pin cleanup passes. The host qualification also completes EXIT0
on the same candidate with the existing canonical serial native control.


The host strict test also passes335 valid requests/20432 exact tokens/259
rejects,including the same empty connections,mixed/reuse/wrap,pipeline,
half-close and all-thread network trace. It records594 host launches,
20432 model graphs,error0,banks1/max_active1,594 publications,eight FREE
slots and complete normal cleanup. Both strict modes use canonical natives,
without the performance listener's test-local TCP_NODELAY setting.
Sources and output are preserved in
[resident strict](../../bench/results/qwen_norm_prepare_resident_correctness_20261010.txt)
and [host strict](../../bench/results/qwen_norm_prepare_host_correctness_20261010.txt).

The [host-replay mechanism profile](../../bench/results/qwen_norm_prepare_profile_20261010.txt)
terminates0. At graph indices5,64,80,128,190 it records760,832,712,712,712
device events versus809,881,761,761,761 in the retained
[norm-tail baseline profile](../../bench/results/qwen_norm_tail_profile_20261010.txt).
Each graph removes49 events;both profiles retain49 MeanOps reduction events.
This supports the intended pointwise preparation mechanism,not a TCP speed
comparison. Sequential replay timing/allocation samples are diagnostic only;
no matched peak-memory or isolated serving benefit is inferred.
The real TCP preflight completes through the admitted unchanged driver.


The real TCP preflight exits0 with15 records,two normal executor exits and
eight exact measured gen1 tokens. Both engines record12 publications,
264 model graphs,one host service launch,error0,banks8/max_active1 and
eight final FREE slots. Full comparison is then started; no preflight
timing is interpreted as a speed result.

### Conversion/square complete TCP result: single-client support only

The [full raw](../../bench/results/qwen_norm_prepare_20261010.jsonl) and
[stdout](../../bench/results/qwen_norm_prepare_20261010.stdout.txt) terminate0:
2621records,40cells,2560requests,83200 exact measured tokens and ten normal
executor starts/exits. Every process records one host service launch,
8580model graphs,error0,banks8/max_active8,264publications,zero queue/
registration failures and eight final FREE slots. All five pairs remain;
there is no failed full cell or post-hoc sample exclusion.
The [analysis](../../bench/results/qwen_norm_prepare_analysis_20261010.txt)
preserves its exact script and independently recomputes every token,
timestamp-derived metric,TPS denominator,prompt/index balance,process order
and counters. Its complete verification passes.

Means below are arithmetic means of five paired blocks. Positive gain means
candidate-minus-baseline TPS or baseline-minus-candidate latency. Nominal
two-sided t95 intervals use df4 and do not adjust for four primary comparisons.

| Primary gen64 | baseline | conversion/square | paired gain | nominal t95 gain |
|---|---:|---:|---:|---|
| clients1 output tokens/s | 285.792567 | 300.975065 | +15.182498 | [+13.444318,+16.920679] |
| clients1 TPOT ms | 3.487801 | 3.309664 | +0.178137 | [+0.158108,+0.198166] |
| clients8 output tokens/s | 682.503354 | 686.412756 | +3.909402 | [-0.701464,+8.520268] |
| clients8 TPOT ms | 11.544272 | 11.485969 | +0.058303 | [-0.019659,+0.136265] |

Single-client throughput improves about5.31% and TPOT falls about5.11%;
both nominal intervals exclude zero. All five concurrent point estimates
favor the candidate,but both concurrent primary intervals cross zero.
This does not establish concurrent benefit,practical equivalence or regression.
The predeclared concurrent retention requirement is unmet: close the candidate
and keep the existing default. Production/test source growth remains0/0;
the four temporary lines are not installed and no fallback/mode is added.

| paired block | clients1 TPS gain | clients1 TPOT reduction ms | clients8 TPS gain | clients8 TPOT reduction ms |
|---:|---:|---:|---:|---:|
| 0 | 15.886689 | 0.184809 | 10.456264 | 0.168944 |
| 1 | 12.759508 | 0.151338 | 1.666569 | 0.029646 |
| 2 | 16.324557 | 0.194232 | 1.632497 | 0.021805 |
| 3 | 15.496145 | 0.182488 | 2.776572 | 0.022519 |
| 4 | 15.445592 | 0.177817 | 3.015108 | 0.048601 |

Secondary gen64 clients1 TTFT4.174313->4.103150ms and
completion223.905766->212.611980ms have nominal positive gain intervals
[0.035905,0.106422] and[10.050011,12.537560]ms.
At clients8,TTFT20.300767->19.945955ms and
completion747.589911->743.562002ms intervals cross zero
[-0.072556,0.782181] and[-0.951369,9.007187]ms.
Gen1/clients1 TPS245.760549->246.672231 and TTFT4.055141->4.038350ms
are inconclusive. Gen1/clients8 TPS274.179040->292.739304 and
TTFT29.011658->27.157760ms have positive intervals
[6.541129,30.579400]tokens/s and[0.896403,2.811393]ms.
These secondary outcomes do not replace the concurrent gen64 primary result;
gen1 has no TPOT.

Warm-cache bootstrap means52.638757s baseline versus54.981165s candidate,
ranges[51.393179,53.690480] and[53.953166,56.217467]s.
Across20 active cells per engine,executor CPU totals0.06/0.08s; five idle
seconds per engine each record0.00s at100Hz. These are coarse process-only
samples,not exact-zero CPU or total host/kernel/energy evidence.
No matched peak-memory benefit is established by the sequential profile.

[Host observations](../../bench/results/qwen_norm_prepare_environment_20261010.txt)
at the collection boundaries retain the same boot and separate SGLang
owner732929/18404MiB. Endpoint400W/100% readings cannot identify per-service
work or within-window co-tenant activity. Post-run cleanup at
2026-10-10T22:30:54UTC finds no owned benchmark/executor,module or BPF pins;
all normal cleanup and final slots also passed. Warm compiler cache,
common temporary TCP_NODELAY,shared RTX5090,short prompts,closed-loop load,
greedy fp16 and client arrival timestamps bound the claim.
This is supporting computation evidence,not isolated bandwidth,total-host
cost,energy,physical-NIC or optimized-serving superiority.
Fresh independent read-only terminal review independently recomputes all
40cells,primary and secondary effects,five paired intervals,oracle and native
counts. It confirms valid/complete execution,supported single-client and
mechanism effects,inconclusive concurrent benefit,supporting engineering
value and computation/workload boundary. The exact candidate/native/frozen
sources,all qualification/strict results,profile and cleanup match.
It finds no invalidating defect and confirms the predeclared close/default
decision. The later batch causal diagnostic is separate from this completed
experiment. No unchanged performance rerun is required.

## Paired KV append copy fusion: real TCP performance experiment

The unchanged goal is worker-free real-model TCP/HTTP serving with improved
throughput/latency. Current default captures HF Qwen computation and advances
eight private graph/KV banks. Its retained RoPE/norm-tail optimizations improve
measured TCP metrics; ordinary and zero-stride BMM batching candidates fail
the fixed greedy oracle. Those closed results cannot answer a different
copy-scheduling mechanism. Prior eager shape attribution associates about13%
of summed diagnostic kernel durations with copies,without identifying that
entire share as KV append or transferring it to device-launched TCP execution.

Hypothesis: compiling the paired key/value cache append together with the
default torch.compile backend reduces copy launch overhead and improves the
current eight-client TCP gen64 throughput/TPOT without changing token outputs,
cache positions or lifecycle. Planned role:supporting engineering evidence,
not a paper RQ or headline superiority. A positive result can retain this
small implementation;contradictory/mixed/inconclusive results preserve the
current default. Existing batching failures do not settle this exact copy
mechanism. A fresh arbitrary matmul retry lacks a qualified implementation;
this copy-only improvement is runnable in the actual service. Correctness
and compiler-mechanism probes are dependencies of the TCP comparison,not
independent paper progress.

Candidate is net4 lines versus byte-exact fc58faa production:
a module-level @torch.compile append_kv executes key.copy_(key_states) and
value.copy_(value_states). SharedLayer.update calls it with the same two
pos:end views and original source tensors. No arithmetic,cast,shape,stride,
KV layout/capacity,position calculation,model,epsilon,queue/module/native
dispatcher,oracle,transport or persistent compiler configuration changes.
The compiler may emit two kernels instead of one,choose a different copy
implementation or specialize on strides. Do not assume fusion or speed.
Use the existing default compiler/cache;no backend/options/precision tuning.
Only the compiled function changes;the returned cache views remain unchanged.

Original Workspace/repository/main only. Base0897b0388d28fb4eb985d9127e5a2100d42b90db,
production/test changes0 at admission. Temporary sources under
/tmp/bpfusion-qwen-pointwise-5mxeys8g:
kv_append/qwen_graph canonical native resident;kv_append_host/qwen_graph_host_launch
canonical serial correctness control;kv_append_perf_baseline/qwen_graph and
kv_append_perf_candidate/qwen_graph identical previously used test-local
TCP_NODELAY native library. Both performance modes are resident eight-bank
engines;the serial host variant is not a performance baseline.
Comparator is the current best bpf-gpu-server implementation,component comparison.
This cannot establish superiority over optimized serving. No service,
installation,new model,mode,scheduler,owner or configuration entity.

Protocol reuses completed norm_prepare fixed evaluation unchanged:
1. qualify_kv_append.py 2 computes independent eager batch1
oracle_cases(64,True) before loading candidate;captures actual device-launched
private graphs;all12 groups/two banks/1536 generated tokens and mismatches.
Full fresh prefill overwrites relevant KV on each group;candidate-selected
tokens drive63 decode steps,no teacher forcing. Any mismatch closes candidate
without performance collection,reference change or unchanged rerun.
2. If all qualify,tests/resident_qwen.py --gen64 --prefill-edges
for BOTH canonical candidate resident and host entries. Each strict protocol
covers335 valid/20432 exact/259 rejects,mixed lengths,reuse,wrap,pipeline,
half-close,5000 empty connections,network trace and cleanup.
3. Actual TCP gen1 preflight through the reused paired driver;completion and
correct tokens only,not speed evidence.
4. Complete five alternating process pairs,rotated gen1/64 x clients1/8
cells,64 measured requests/cell,four prompts16 times/cell:
40cells2560 requests83200 exact tokens. Preserve every paired block,
negative/inconclusive/failure row,bootstrap and process CPU cost.
Inherited ready-or-exit,shutdown/request60s handling,common transport,
warmup/exclusions and metric definitions remain unchanged.

Commands from /workspaces/repository with existing
HF_HOME=/workspaces/.cache/huggingface:
python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_kv_append.py 2
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/kv_append/qwen_graph
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/kv_append_host/qwen_graph_host_launch
python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/kv_append_bench.py --gen 64 --preflight --output bench/results/qwen_kv_append_preflight_20261010.jsonl --resident-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/kv_append_perf_baseline/qwen_graph --host-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/kv_append_perf_candidate/qwen_graph
Full removes --preflight and writes qwen_kv_append_20261010.jsonl.
Ordinary stdout/qualifier/strict raws under bench/results/qwen_kv_append*.
Freeze exact sources and ordinary CPU analyzer before GPU execution.
CPU analyze_kv_append.py reconstructs all tokens/timestamps/TPS,
cell/prompt balance,engine order,process counters and nominal intervals.

After qualification,profile_kv_append.py captures one bank and replays five
model graphs (5,64,80,128,190),with installed PyTorch profiler. Compare
device event count/copy kernel identities against retained norm-tail profile
using the same default model math,not standalone replay latency as TCP speed.
Generated installed-compiler source can show whether both writes occur in
one kernel. This is mechanism evidence folded into the same experiment;
no guaranteed reduction/fusion or physical bandwidth claim.

Primary gen64 TPS and TPOT measured at clients1/8 as separate nominal paired
df4 t95 effects over five process-block means;positive gain=candidate-minus-
baseline TPS or baseline-minus-candidate latency. Retain only when both
concurrent primary outcomes support improvement and no primary outcome
supports regression. Single-client inconclusive effects remain separate.
TTFT,completion,gen1,bootstrap,coarse process CPU and capture allocations
cannot substitute for the primary result. The formulas/definitions and
transport conditions are the already published component protocol in0021.

Before/after host boot and GPU compute residency preserve other SGLang owner,
with no induced isolation. Terminal own-process/module/map cleanup.
Warm cache,shared GPU,one small model/stack,finite short prompts/closed loop,
temporary common TCP_NODELAY and no total-hostCPU/energy/realNIC/external
optimized baseline limit interpretation. Exact numerical and TCP lifecycle
checks are necessary,not sufficient for general service correctness.
Fresh plan review before execution;fresh independent terminal result review.
If validated effects support retention,promote only the net4 lines and rerun
canonical default-cache/default-network strict regression before push.
If rejected/unretained,production/test growth0 and preserve raw honest result.


Fresh independent plan review:ADMIT,no invalidating scientific/execution
defect. Actual candidate/frozen source/native identity,independent oracle,
full fixed TCP comparison/analysis and retention scope checked. Copies may
not fuse:top15 profiler names alone cannot prove this;use generated source
or full physical records,otherwise report mechanism unknown. The analyzer's
ordinary collection-commit reference file contains
0897b0388d28fb4eb985d9127e5a2100d42b90db and is frozen below for public replay.
API references:[torch.compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
and [copy_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.copy_.html).
No GPU execution preceded this admission.

Real device-bank qualification completes12 groups/1536 exact tokens,
each with one host probe launch/128 model graphs/error0. Independent CPU
checks all64-token sequences and group identities against the frozen oracle.
Installed generated static/dynamic wrappers each invoke the copy kernel
twice,once per original key/value target. Thus the expected horizontal
single-kernel fusion is not realized. This does not change the candidate
or evaluation:the admitted full TCP comparison can still test whether its
different copy implementation has useful effects. Do not claim fewer KV
launches or a fusion mechanism from this candidate. Generated source is
frozen with the adapter. Before profiler execution,add only a full grouped
device-record print to the existing profile output to preserve non-top15
names;capture and replay are unchanged.

The first canonical resident strict invocation fails before executor startup:
strace reports exec Permission denied because these newly written temporary
entry files lacked executable bits. This is an adapter preparation error,
not a model/kernel/host failure;llm_head remains0 and test cleanup removes
its maps/module. Preserve the failed raw qwen_kv_append_resident_regression_20261010.txt.
Restore ordinary0755 on the four temporary Python entries and rerun the same
strict command into resident_regression_repaired_20261010.txt. No source,
oracle,model,transport,owner or persistent permission boundary change.

Both canonical strict modes complete335 valid requests/20432 exact tokens/
259 rejects,594 publications,zero graph errors and cleanup. Resident has
one host launch/eight banks/max_active8;host control594 launches/one bank/
max_active1. Network trace confirms no executor accept/read/send after ready.
The profile completes all five shapes with total event counts
809,881,761,761,761,identical to retained norm-tail baseline;each contains48
triton_poi_fused_copy_copy__0 physical kernels. Generated wrappers show
two launches per layer. Paired kernel fusion/launch reduction is contradicted;
different copy implementation remains the actual intervention. Diagnostic
replay timings are not a TCP benefit and do not replace the planned comparison.

Real TCP preflight completes15 records/eight exact measured gen1 tokens,
two normal resident executor exits. Each has one host launch/264 model
graphs/error0/eight banks/max_active1,12 publications and eight FREE slots.
No failure row. This establishes actual collection,not a speed result.
Start the unchanged full five-pair40-cell comparison.

### Compiled KV copies complete TCP result: no concurrent retention

[Full raw](../../bench/results/qwen_kv_append_20261010.jsonl) and
[stdout](../../bench/results/qwen_kv_append_20261010.stdout.txt) terminate0:
2621 records,40 unique cells,2560 measured requests,83200 exact tokens,
ten normal executor starts/exits and no failure row. All five alternating
pairs and all samples remain. Every executor records one host launch,
8580 model graphs,error0,eight banks/max_active8,264 publications,zero
queue/registration failures and eight final FREE slots.
The frozen [CPU analysis](../../bench/results/qwen_kv_append_analysis_20261010.txt)
independently recomputes all oracle tokens,timestamps,TPS denominators,
prompt/cell balance,process order/counters and uncertainty.

Means below are arithmetic means of five process-block pairs. Gains are
candidate-minus-baseline TPS or baseline-minus-candidate latency;nominal
two-sided paired t95 usesdf4 and has no four-comparison multiplicity adjustment.

| Primary gen64 | baseline | compiled copies | gain | nominal t95 gain |
|---|---:|---:|---:|---|
| clients1 output_tokens_per_s | 285.989446 | 288.794539 | +2.805093 | [+1.020099,+4.590087] |
| clients1 tpot_ms | 3.484949 | 3.450561 | +0.034388 | [+0.013626,+0.055150] |
| clients8 output_tokens_per_s | 680.116843 | 686.034127 | +5.917284 | [-7.932435,+19.767004] |
| clients8 tpot_ms | 11.586925 | 11.481702 | +0.105222 | [-0.136919,+0.347364] |

Single-client TPS improves about0.98% and TPOT about0.99%;both nominal
intervals exclude zero. Concurrent mean estimates favor the candidate,but
both intervals cross zero;the first pair is worse. This establishes neither
concurrent benefit,equivalence nor regression. The predeclared concurrent
retention requirement is unmet:close this candidate,keep default source
byte-exactfc58faa and preserve all raw. Production/native/test growth0.
Do not rerun an unchanged five-pair matrix to seek significance.

The actual intervention is replacement of48 ATen KV copies by48 compiled
copy kernels per model graph,not paired single-kernel fusion. Five profile
totals and generated wrappers independently contradict fewer KV launches.
The original reduction hypothesis is contradicted;single-client copy-
implementation improvement has narrow support,concurrent effects are
inconclusive. No bandwidth or unique causal attribution follows.

Warm-cache bootstrap means53.178692s baseline versus57.032212s candidate,
ranges[51.843666,54.711229] and[55.869288,57.584775]s. Executor-only CPU totals
across active cells are0.06s each;idle totals0.01s each,at100Hz. These do not
prove exact-zero CPU,total-host cost or energy. Sequential one-bank capture
allocated1089993728/reserved1518338048bytes,matching the old norm-tail
diagnostic;it is not a matched peak service memory result.

Both strict [resident](../../bench/results/qwen_kv_append_resident_regression_repaired_20261010.txt)
and [host](../../bench/results/qwen_kv_append_host_regression_20261010.txt)
regressions remain complete. The [initial entry-mode failure](../../bench/results/qwen_kv_append_resident_regression_20261010.txt)
is preserved as a before-startup adapter fault,not a performance cell. No
successful performance cell is excluded;ordinary0755 restoration changes no oracle or model.
[Qualification](../../bench/results/qwen_kv_append_qualification_20261010.txt),
[profile](../../bench/results/qwen_kv_append_profile_20261010.txt),
[preflight](../../bench/results/qwen_kv_append_preflight_20261010.jsonl),
[frozen source](../../bench/results/qwen_kv_append_adapter_20261010.txt)
and [before/after host observations](../../bench/results/qwen_kv_append_environment_20261010.txt)
preserve reproduction. Terminal cleanup finds no owned executor/driver/module/
pins;boot and other SGLang732929/18404MiB remain unchanged.

This is a supporting component comparison on one shared RTX5090,with warm
compiler cache and common temporary TCP_NODELAY. It is not optimized-serving
superiority,isolated throughput,totalCPU/energy,physicalNIC,text processing
or a whole-goal result. The next engineering discriminator is an actual
single kernel performing both original strided copies,without changing
floating-point model arithmetic,rather than recompiling the same two launches.

Fresh independent terminal result review:valid/complete,no invalidating
defect. It recomputes all2621 records/40cells/2560requests/83200 exact tokens,
all paired effects/secondary costs,and independently checks actual/frozen
source,native library identity,oracle,device probe,strict raws and cleanup.
Launch-reduction mechanism is contradicted;single-client gains have narrow
nominal support,concurrent primary effects remain inconclusive. Retention
is unmet:close candidate,keep default. Supporting component evidence only.


### Actual paired KV bit-copy candidate: plan and admission review

## Single-kernel bit-preserving paired KV append

Under the unchanged worker-free real-model TCP/HTTP throughput/latency goal,
the completed default-compiler paired-copy candidate stays48 KV kernels/model
graph. Its five-pair comparison has narrow single-client support but
inconclusive concurrent primary effects and does not meet retention.
No unchanged retry is proposed. A distinct single kernel can eliminate one
launch per layer while preserving both original copies and model arithmetic.
This is supporting engineering evidence;qualification/profile are dependencies
of one real TCP performance experiment,not new paper RQs or contributions.

Hypothesis:one strided bit-copy kernel replacing two separate KV copies
improves concurrent gen64 throughput/TPOT with unchanged greedy tokens,
cache contents/positions and request lifecycle. Fewer kernels alone is not
the primary success criterion;it may be slower,especially at prefill.
Positive effects can retain the implementation;negative/mixed/inconclusive
effects preserve default. This directly tests the missing launch-reduction
mechanism versus the closed compiler candidate. The best qualified default
is the main component comparator;the serial host path remains a correctness
control,not an external baseline. No optimized-serving superiority claim.

Scope:existing Workspace/main. Current production remains byte-exactfc58faa.
Net24 source lines in a temporary candidate;no native/module/queue/test,
model/precision/oracle/transport change. Triton is already installed/used by
retained pointwise compilation;no installation/helper/service/mode/entity.
Use the ordinary installed jit/load/store path,following
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html .
No autotune/backend/num_warps/options override. Fixed128-element tile is
kernel geometry matching this model's two64-wide KV heads per position,
not a service runtime tuning setting;larger prefill spans multiple tiles.
No tuning sweep or claimed optimal launch geometry.

Kernel copy_kv maps flattened B,H,N,D logical indices to original source
strides for key/value and target strides for both. SharedLayer passes the
same pos:end target views,so their data pointers already include the cache
offset. Key/value destination strides coincide because original allocation
uses values=zeros_like(keys);both are independently allocated. Input strides
may differ and are passed separately. In-place stores target the original
buffers;returned prefix views are unchanged. Actual model tensors arefp16.
Pointers are reinterpreted as uint16;loads/stores perform no floating
arithmetic or value conversion. Mask x<TOTAL covers partial final tiles.
This is a source argument requiring actual oracle/lifecycle validation,
not an assumption that finite passed prompts prove general correctness.

Only this copy implementation differs. Temporary paired_kv/qwen_graph and
paired_kv_host/qwen_graph_host_launch use canonical native libraries.
paired_kv_perf_baseline/qwen_graph is byte-exact current production;
paired_kv_perf_candidate/qwen_graph is byte-exact candidate. Both performance
entries use the same existing eight-bank temporary TCP_NODELAY native library.
Host control is serialized for correctness only. Other SGLang owner,
original Workspace/PVC/session/tools/setup_env.sh are preserved.

Reuse the already complete fixed protocol:
- qualifier computes independent eager-fp16 batch1 oracle BEFORE candidate,
12 groups/two device-launched private banks/1536 generated tokens;
fresh prefills overwrite relevant KV,then63 candidate-selected decode steps.
Any mismatch closes the unchanged candidate without performance collection.
- Both canonical strict resident/host regressions:335 valid requests,
20432 exact tokens,259 rejects each,mixed lengths,reuse,wrap,invalid-then-valid,
pipeline,half-close,5000 empty connections,network trace and cleanup.
- Actual TCP gen1 preflight,two resident engines/eight exact measured tokens.
- All five alternating process pairs with rotated gen1/64 x clients1/8 cells,
64 requests/cell,four prompts16 times/cell:
40cells2560 measured requests83200 exact tokens,ten normal exits.
Inherited metrics/exclusions,ready-or-exit and request/shutdown60s behavior
remain unchanged. All failed/negative/inconclusive rows remain. No oracle,
timeout or evaluation change during qualification or performance.

The qualifier/profiler ordinarily derive capture/probe code from the exact
candidate,as in prior experiments. Triton jit needs inspectable real source:
write the generated model to an ordinary OS-temp .py file and execute with
runpy.run_path instead of compile/string exec. Frozen adapter and raw commands
preserve this necessary loader difference. It changes no model/capture/probe
operation. CPU syntax checks happen before GPU execution.

From repository root with existing HF_HOME=/workspaces/.cache/huggingface:
python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_paired_kv.py 2
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/paired_kv/qwen_graph
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/paired_kv_host/qwen_graph_host_launch
python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/profile_paired_kv.py
python3 /tmp/bpfusion-qwen-pointwise-5mxeys8g/paired_kv_bench.py --gen 64 --preflight --output bench/results/qwen_paired_kv_preflight_20261010.jsonl --resident-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/paired_kv_perf_baseline/qwen_graph --host-executor /tmp/bpfusion-qwen-pointwise-5mxeys8g/paired_kv_perf_candidate/qwen_graph
Full removes --preflight and writes qwen_paired_kv_20261010.jsonl.
Ordinary qualifier/strict/profile/stdout/analysis/environment raws share
qwen_paired_kv prefix. Freeze all actual source/analyzer/driver and collection
HEAD before GPU execution. analyze_paired_kv.py checks tokens,timestamps,
TPS denominator,prompt/cell balance,engine order,counters and all intervals.

Profile one bank/five graph indices5,64,80,128,190 with installed profiler,
preserving ALL grouped device names/counts. Expect24 copy_kv kernels and
total CUDA events785,857,737,737,737 versus retained default809,881,761,761,761.
Confirm empirically;do not infer from source/kernel names alone. Host replay
timings/allocations are diagnostics,not serving speed or matched peak memory.
No custom checker framework or independent toy bandwidth benchmark.

Primary gen64 TPS/TPOT at clients1/8 are separate nominal paired df4 t95
effects from five process-block means;positive gain=candidate-minus-baseline
TPS or baseline-minus-candidate latency. Retain only when both concurrent
primary effects support improvement and no primary effect supports regression.
Report single-client inconclusive effects separately. Secondary TTFT,
completion,gen1,bootstrap,executorCPU or fewer kernels cannot substitute.
All metrics/formulas are the published component protocol in0021.

Fresh plan/independent terminal result review;before/after boot/residency,
own-process/module/map cleanup. If retention is supported,promote only
this source change and pass canonical default-cache/default-network complete
strict regression before commit/push. If not,production/native/test growth0.
Do not repeat unchanged full matrices to seek significance.

Shared RTX5090/warm cache/one small model stack/finite short prompts/closed
loop/common temporary TCP_NODELAY/coarse processCPU limit interpretation.
No total-hostCPU,energy,isolated bandwidth,physicalNIC,text processing or
external optimized-serving win. A smaller kernel count does not by itself
establish improved throughput/latency or complete the user goal.

Plan review round1 found one execution defect before GPU execution:
runpy changes argv[0] and __file__,so original source chooses absent
/tmp/build/qwen_graph.so. Correct only generated-source native library
binding:qualifier uses the existing graph_bank_probe/qwen_graph.so;
profiler uses canonical /workspaces/repository/build/qwen_graph.so.
Actual candidate entries,model math,oracle,capture/probe,protocol and
performance/native identity are unchanged. Re-freeze adapters before run.

Frozen [adapter/source/analysis](../../bench/results/qwen_paired_kv_adapter_20261010.txt). Collection parent commit 6ad1e1122ccfa2b0ce77f7b94d8951aac0520d66. No GPU execution yet.

Plan review follow-up1:ADMIT. Independent inspection confirms both explicit
existing native paths,syntax,frozen adapter,byte-exact actual entries and
collection commit. Sole runpy loader blocker repaired before execution;
all prior mechanism/oracle/fairness/count/retention checks stand.
No GPU/model/service/test was executed by reviewer. Begin qualification.

Device-bank qualification completes12 groups/two banks/1536 exact tokens,
each128 model graphs/one host launch/error0. All candidate-selected decode
sequences pass the unchanged independently constructed eager-fp16 reference.
No loader fault recurs. Proceed to both canonical strict TCP regressions.

Both canonical strict modes pass335 valid requests/20432 exact tokens/
259 rejects,594 publications,zero graph errors and cleanup. Resident uses
one host launch/eight banks/max_active8;host correctness control uses594
host launches/one bank/max_active1. All executor threads trace no network
accept/receive/send after ready and all eight final queue slots are FREE.

Actual one-bank profile confirms24 copy_kv physical kernels in each of five
model graphs. Total device event counts785,857,737,737,737 versus retained
default809,881,761,761,761:24 fewer events per graph. The single-kernel
paired-copy mechanism engages. Host replay timings are diagnostic only;
the actual TCP experiment determines throughput/latency effects.

Real TCP preflight completes15 records/eight exact measured gen1 tokens,
two normal resident executor exits. Each uses one host launch/264 model
graphs/error0/eight banks/max_active1,12 publications/eight final FREE slots.
No failure row. Begin unchanged full five-pair40-cell TCP comparison.

### Actual paired KV kernel complete result: mechanism engages, performance inconclusive

[Full raw](../../bench/results/qwen_paired_kv_20261010.jsonl) and
[stdout](../../bench/results/qwen_paired_kv_20261010.stdout.txt) terminate0:
2621 records,40 unique cells,2560 measured requests,83200 exact tokens,
five alternating process pairs,ten normal engine starts/exits,no failure row.
Every executor records one host launch/8580 model graphs/graph_error0,
eight banks/max_active8,264 publications,zero queue/registration failures
and eight final FREE slots. No measured sample/block was removed or rerun.
The frozen [CPU analysis](../../bench/results/qwen_paired_kv_analysis_20261010.txt)
recomputes every token,timestamp,TPS denominator,prompt/cell balance,
process ordering/counters and effect interval.

Means are arithmetic means of five paired process blocks. Positive gain is
candidate-minus-baseline throughput or baseline-minus-candidate latency.
Separate nominal two-sided paired t95 intervals use df4,with no adjustment
for the four primary comparisons.

| Primary gen64 | baseline | paired KV kernel | gain | nominal t95 gain |
|---|---:|---:|---:|---|
| clients1 output_tokens_per_s | 286.237192 | 290.222213 | +3.985021 | [-0.101312,+8.071355] |
| clients1 tpot_ms | 3.481845 | 3.434050 | +0.047795 | [-0.001202,+0.096793] |
| clients8 output_tokens_per_s | 682.401193 | 688.183273 | +5.782080 | [-6.977520,+18.541680] |
| clients8 tpot_ms | 11.542205 | 11.451740 | +0.090466 | [-0.134973,+0.315905] |

All four primary intervals cross zero. Single-client mean throughput
286.24->290.22 tokens/s and TPOT3.482->3.434ms favor the candidate;
eight-client mean throughput682.40->688.18 and TPOT11.542->11.452ms also
favor it,but one process pair worsens both client regimes. This supports
neither a primary performance improvement nor equivalence or regression.
The predeclared concurrent retention condition is unmet:close candidate,
keep production byte-exactfc58faa,production/native/test growth0.
Do not repeat the unchanged matrix to seek significance.

The actual mechanism succeeds:each profiled graph has24 copy_kv kernels
instead of48 KV-copy kernels,and total device events decline by24 in all
five profiled shapes. This confirms single-kernel paired copying;it does
not establish physical bandwidth savings,unique causal attribution or
TCP performance. The throughput/TPOT hypothesis remains inconclusive.

Warm-cache bootstrap means52.872640s baseline versus58.619656s candidate,
ranges[52.456020,53.405419] and[57.117530,61.641073]s.
Active-cell executor CPU totals0.07s versus0.06s;idle totals0 each at100Hz.
Zero accounting ticks are not exact-zero work,total-host CPU or energy.
One-bank capture allocated1089993728/reserved1518338048bytes and held
57735680bytes of logits,matching retained diagnostic allocations.
This is not a matched peak service-memory result.

Both canonical strict [resident](../../bench/results/qwen_paired_kv_resident_regression_20261010.txt)
and [host control](../../bench/results/qwen_paired_kv_host_regression_20261010.txt)
pass335 valid requests/20432 exact tokens/259 rejects.
[Qualification](../../bench/results/qwen_paired_kv_qualification_20261010.txt),
[profile](../../bench/results/qwen_paired_kv_profile_20261010.txt),
[real preflight](../../bench/results/qwen_paired_kv_preflight_20261010.jsonl),
[frozen adapter](../../bench/results/qwen_paired_kv_adapter_20261010.txt)
and [host environment](../../bench/results/qwen_paired_kv_environment_20261010.txt)
preserve commands/source/reference/costs. The runpy loader defect was caught
and repaired before execution;no qualification/strict/performance failure
was hidden. All subsequent invocations terminate0 with unchanged evaluation.

Terminal own executor/driver/module/map pins are absent. Before/after boot
04dc0087-5ccc-4c46-9997-b9a92f0e5167 and other owner SGLang732929/18404MiB
remain unchanged. Shared-GPU/warm-cache/one model+stack/finite short prompts/
closed-loop/common temporary TCP_NODELAY/coarse process CPU limit the result.
No isolated speedup,optimized-serving superiority,real NIC,text/API,
total-host CPU/energy or whole-goal completion follows.

This is supporting component evidence:the proposed launch-reduction
mechanism engages,but concurrent performance is inconclusive. Further
unchanged copy experiments add little. The next material discriminator is
a faithful token-counted comparison with an official optimized serving
implementation,and independent per-bank scheduling remains an architectural
lead rather than a tested candidate. Existing bench/run_http.sh uses broad
pkill and tools/llm_http_client.py counts nonempty SSE text chunks as tokens;
neither can serve unchanged as a shared-host matched token-latency comparison.
Official [vLLM completion protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/)
accepts token-ID prompts and can return delta token IDs,avoiding that metric
defect. Current base and existing /workspaces/.venv-ml imports lack vLLM;
live vLLM0.31.0 requires torch2.13.0/transformers<5.18,so preserve the
qualified torch2.14.1/transformers5.19 environment. These are read-only
feasibility observations;no external serving run or package installation
occurred during this collection,and no competitive result is claimed.

The complete secondary analysis also shows nominal gen1/eight-client support:
throughput gain+10.301454 tokens/s,[+2.645129,+17.957778],and TTFT/completion
reduction1.082429ms,[+0.282247,+1.882610]. Other secondary intervals cross zero.
These are secondary observations with no multiplicity adjustment;they do not
substitute for gen64 primary retention and do not justify changing default.

Fresh independent terminal result review:valid/complete,no invalidating
defect. It recomputes all2621 records/40cells/2560requests/83200 exact tokens,
primary and secondary effects,costs,order/counters and final FREE slots;
actual/frozen native/source/oracle/adapter identity and profile/strict/cleanup
also match. It independently reads39 printed expected/actual pairs in each
strict raw;the remaining aggregate strict correctness relies on unchanged
test assertions and successful exit,not a claim that every strict token is
printed. Single-kernel mechanism is supported,gen64 performance inconclusive,
supporting component evidence only. Close candidate,keep default,no unchanged
rerun;official optimized-serving comparison remains a separate next experiment.
