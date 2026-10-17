# Upstream batch-invariant serving: exact finite fidelity, no throughput gain

Observed2026-10-15UTC in the original yunwei356/bpfusion-dev Workspace/main,
base3cb7685f368195a1d398ca9639a143df99e64a28.
[Plan](../../bench/results/batch_invariant_serving_20261015/plan.md),
[plan review](../../bench/results/batch_invariant_serving_20261015/plan-review.md),
[result review](../../bench/results/batch_invariant_serving_20261015/result-review.md)
and [raw archive/replay](../../bench/results/batch_invariant_serving_20261015/)
retain the experiment. No production source or default is changed.

## Question and intervention

The retained question is compute-matched Linux TCP/eBPF/shared-queue/resident-GPU
serving without a steady userspace request worker, improving throughput/latency.
This supporting experiment asks whether upstream batch-invariant FP16 computation
can supply independently checked ordinary outputs and a fair loop/native delivery
comparison. It invents no formal paper RQ or determinism technique.

Finding0058 exposed the finite reference-pool boundary of ordinary default-mode
runs. Here each expected sequence comes from official CPU-scheduled LLM.generate
before observing bpf-gpu-server. Enable only upstream VLLM_BATCH_INVARIANT=1 in a private
test environment. This selects upstream operation/environment behavior; it changes
the compute mode from earlier default-mode results. Loop and efficient timed-epoll
native TCP share that same mode, graphs, weights, precision and existing timer.
Official generation is a correctness control, not a performance baseline.

The pinned source's BF16 tuned breakable-graph default does not apply to this
FP16 constructor. Private bootstrap changes only its root path and startup
mode assertion/settings print. Capture, dispatcher, shared transport and model
sources are unchanged. Effective mode settings include CUBLAS_WORKSPACE_CONFIG
:16:8, CUBLASLT_WORKSPACE_SIZE=1 and VLLM_USE_AOT_COMPILE=0, selected by upstream.
No local model kernel, upstream patch, clock/driver or timer change is introduced.

## Independent reference and actual execution

Each model first generates ten serial64-token references through official
LLM.generate with temperature0, ignore_eos=True, detokenize=False. Subsequent
gen1/64 calls at actual full widths1..8, forward/reverse mixed input order,
must equal the single frozen sequence or its prefix. Each model completes
394requests13120tokens with no disagreement; two models total788requests26240tokens.
The ten inputs are the four ordinary prompts plus protocol lengths1/15/16/17/32/64.
No accepted-output union, serving-derived reference or post-observation repair
is used. This establishes finite exact token fidelity, not logits/task quality.

Ordinary Qwen real preflight passes both transports:64 measured requests2080tokens
plus16 checked gen1 warmups. Both models and both transports then pass each
622-request protocol suite, including259 intentional rejections and5000 empty
connections. Each suite checks22224tokens against both the official references
and independently advanced upstream graph/KV/history replay, with batches1..8.
Four suites total2488requests88896tokens. System-call checks verify no steady
loop userspace accept/receive/send handling and the expected initial io_uring_enter;
native actually engages its TCP mechanism. Linux/kernel/timer CPU work remains.
The existing bounded reply-diagnostic truncation is distinct from model trace
overflow; raw counters retain that distinction.

The fresh independent review judges this finite comparison valid, official
fidelity supported, throughput superiority inconclusive, and research value
supporting with a mechanism/workload boundary. The untraced full matrix
completes20executors80cells:2560 measured requests83200
output tokens plus160 checked warmups. Both model drivers and fixed analyses
exit0; wrapper terminal is05:22:02.721225UTC. All20 outputs match the independently
frozen sequences exactly. No failed/excluded/replaced serving cell, retry,
reference/timeout/metric change or profiler is used. Models run Qwen then Smol;
within model, five paired blocks alternate transport order and rotate cell order.

## Performance and uncertainty

Primary is gen64/c8 client output token/s. Intervals below are nominal paired
t95df4 across five process blocks, not pooled requests or familywise intervals.

| Model | Loop token/s | Native token/s | Loop minus native [t95] |
|---|---:|---:|---:|
| Qwen2.5-0.5B |1204.243|1213.613|-9.371 [-18.755,0.014]|
| SmolLM2-135M |2304.157|2303.451|+0.706 [-56.624,58.037]|

Both primary intervals cross zero: throughput superiority and equivalence are
unestablished. All five Qwen signed throughput differences are negative; the
near-zero upper endpoint still cannot be rounded into an exclusion of zero.
All block effects,16 metric rows and52 signed secondary/primary contrasts remain
in the fixed analysis and fresh review; favorable secondary rows do not replace
the primary. CPU rows are known-worker/process brackets, not total service CPU. Smol gen64/c1
known-worker CPU is adverse+1.279ms/request[0.327,2.231]; this remains partial
accounting and is not hidden by the inconclusive primary.

| Model | Gen1/c1 TTFT loop/native, ms | Gen64/c8 TTFT loop/native, ms | Long TTFT loop minus native [t95], ms |
|---|---:|---:|---:|
| Qwen2.5-0.5B |6.589/6.741|11.538/7.992|+3.547 [2.635,4.458]|
| SmolLM2-135M |3.645/3.597|6.047/4.976|+1.071 [0.147,1.995]|

Long-output concurrent TTFT is adverse in all five blocks/model. Single-request
TTFT contrasts cross zero. Outliers and all completion/TPOT/CPU results remain
available. The absolute single-request times do not justify making this mode
the low-latency default. No new default-mode condition was measured, so this
experiment cannot quantify mode overhead or compare earlier default results
as a paired causal estimate. Differing batches/history prevent exclusive
network, GPU-service or admission-cost attribution.

## Scope, selection and restoration

Use this evidence for the finite same-mode fidelity and transport comparison.
It does not explain0058's output failure or qualify default-mode ordinary
trajectories, eager-HF parity, arbitrary models/backends, text/OpenAI APIs,
total CPU/energy, production p99 or continuous host isolation. Upstream mode
is reused functionality, not bpf-gpu-server novelty. No speedup or default promotion
follows; the original source remains selected.

Pinned private runtime is vLLM0.31.0, Torch2.13.0+cu129, Transformers5.17.0,
NumPy2.3.5, Triton3.7.1 and FlashInfer0.7.0.post1, with libbpf1.6.3/NVCC13.0.88.
Checkpoint revisions match0055/58. Host snapshots record the same boot/400W
setting and56 unchanged selected historical warnings. Active GPU ownership maps
to the original Pod/container. After snapshot05:23:18.874653UTC is76.153428s after
terminal; sparse samples do not establish continuous isolation or harmlessness.

Review caught a list/tuple comparison defect before any GPU observation; it was
fixed without changing expected IDs. After execution, stale saved diff files
were regenerated from actual executed sources against saved originals. Actual
scripts/raw metadata already contained the repairs; no result/source was edited.
Execution notes retain these corrections.

All owned test processes are absent. Private runtime,16 build entries and libbpf
dependencies are removed. Original eight setup/build paths, base packages and
Coder PID/start times remain equal; module and pins are absent. Five post-cleanup
CPU analysis replays reproduce exit0 and identical stdout/stderr.
[Restoration](../../bench/results/batch_invariant_serving_20261015/final_restoration.json)
retains observations. This comparison is closed; the broader serving/performance
objective remains active.
