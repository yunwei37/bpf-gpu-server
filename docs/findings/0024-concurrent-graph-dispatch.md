# 0024: concurrent whole-request graph dispatch control

## Question and admission

The user asks for throughput/latency improvement and a reproducible matched comparison of Linux TCP/HTTP -> eBPF/shared queue -> resident real-model inference -> kernel TCP TX. This repository has no paper RQ, thesis or evaluation contract. The supporting engineering question is: does keeping whole-request service dispatch resident improve client TTFT and output throughput when both policies have the same eight private graph/KV banks and GPU-controlled decoding?

Finding0020 measures serial dispatch and finding0023 measures serial-to-concurrent execution. Neither can isolate dispatch against the present concurrent path: its old host control has one bank. Reanalysis cannot restore the missing concurrent host condition. This comparison closes that confound; it is a component ablation, not an external main serving baseline or a headline novelty result. A positive effect supports this local control policy. Null/negative or mixed effects bound its latency/throughput benefit and direct the next effort toward stronger upstream-backend/optimized-server/physical-NIC evidence rather than repeated unchanged dispatch runs.

The more consequential alternative is a matched optimized serving comparison using a qualified upstream megakernel. Current MPK and Plow SM120 qualifications have failed exact output controls; the other candidates need qualification and online lifetime adaptation. Correcting the already-qualified graph control is the strongest currently runnable causal discriminator without another model-math tuning campaign. Its result cannot settle competitiveness or establish the entire service goal.

## Matched implementation

Both modes capture the same BF_LLM_SLOTS graph banks using the same Python source, shared read-only exported Qwen2.5-0.5B fp16 weights, eager HF batch1 computation, device argmax, private input/KV/output layout, graph catalogue, admission loop, FIFO retirement/reclamation and eBPF/kernel TCP path. The control returns after each published DONE, and its host synchronizes and resumes the same service graph; decode remains GPU-tail controlled, with no host token loop.

Sampling now moves an unfinished request from phase1 (child output waiting to be consumed) to phase3 (next model graph waiting to be launched). A host return while retiring another slot must preserve that distinction; otherwise re-entry would read stale output as a new token. The resident path launches phase3 in the same iteration, preserving its computation sequence. Host mode has eight banks, so the earlier single-bank admission restriction is removed. No protocol, page ABI, model arithmetic, precision, checkpoint or persistent runtime setting changes.

Correctness precedes measurement: the existing gen64/prompt-edge TCP/HTTP regression must pass for both modes, including 335 valid requests/20432 exact output tokens, 259 rejections, eight concurrent clients, mixed gen1/gen64 lengths, immediate reuse, ring wrap, ordered pipeline, half-close and all-thread network tracing. Both report banks8/max_active8, graph_error0 and 20432 model steps; host service launches594, resident1. Exact HF oracle and request/shutdown timeouts remain fixed.

## Workload and interpretation

Reuse bench/resident_dispatch.py and the four existing real prompts with independently computed eager-HF fp16 reference tokens. Use gen1/gen64 and closed-loop client concurrency1/8; both modes progress private batch1 requests, not tensor-batched matrix operations. Five paired process blocks alternate mode order, rotate the same cell order within a pair and serve64 measured requests/cell (16 per prompt), plus eight excluded gen1/gen64 warmups/process.

The [NVIDIA metric definitions](https://docs.nvidia.com/nim/benchmarking/llm/metrics) supply client TTFT, TPOT excluding the first token, completion latency and aggregate output TPS over first-send/last-token-arrival. Primary concurrent gen64 effects are host-minus-resident mean TTFT and resident-minus-host TPS, assessed separately. Report every pair and nominal two-sided t95 intervals, df4, using the [NIST critical value](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm). Positive intervals support a local benefit; negative intervals contradict it in that cell; crossing zero is inconclusive, not equivalence. Clear tradeoffs are mixed support. Gen1, one-client cells, completion and TPOT remain diagnostics rather than hidden vetoes. No familywise superiority or robust P99 claim is made from five pairs.

Connection/thread creation, model load/capture, warmup, GPU queries and shutdown precede or follow the measurement windows. Multiple tokens completed in one recv share an arrival timestamp. Bootstrap duration is recorded separately. Executor process CPU includes its threads but excludes kernel TX/softirq. Raw global /proc/stat counters additionally record user+nice+system+irq+softirq CPU work in approximately the same cell window, including client threads, kernel work and unrelated host workloads. They are descriptive whole-host observations, not an attribution of service cost or energy savings.

Both use the canonical default listener buffering: no temporary TCP_NODELAY build or transport tuning. Both clients use the existing common TCP_NODELAY setting. Prior NODELAY runs are historical evidence, not numeric controls for this collection. Another project's SGLang workload stays in place on the shared RTX5090; record coarse GPU state and host process residency/boot/Xid observations. Alternating pairs reduce but cannot eliminate temporal confounding. No isolated speedup, GPU-ready latency, physical-NIC, text/tokenization, integrated energy or optimized-serving claim follows.

## Execution and completeness

Build with make qwen-graph qwen-graph-control in the original bpfusion-dev Workspace/main. Native default binaries, capture code, weights and oracle are identical across preflight/full run. The runner waits for readiness or process exit rather than the obsolete60-second capture cap; record this before execution because eight banks take longer to capture. Original request and shutdown timeouts are unchanged.

Commands:

```sh
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph_host_launch
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph
HF_HOME=/workspaces/.cache/huggingface python3 bench/resident_dispatch.py --gen 64 --preflight --resident-executor ./build/qwen_graph --host-executor ./build/qwen_graph_host_launch --output bench/results/concurrent_graph_dispatch_preflight_20261011.jsonl
HF_HOME=/workspaces/.cache/huggingface python3 bench/resident_dispatch.py --gen 64 --resident-executor ./build/qwen_graph --host-executor ./build/qwen_graph_host_launch --output bench/results/concurrent_graph_dispatch_20261011.jsonl
python3 bench/resident_dispatch.py --analyze --output bench/results/concurrent_graph_dispatch_20261011.jsonl
```

Preflight has four exact measured gen1 requests/mode following the same eight warmups. Full completion requires40 unique cells,2560 measured requests,83200 exact measured tokens, ten normal process exits, every index0..63 present once/cell and all five pairs retained. Each full process serves264 requests/8580 model graphs, reports banks8/max_active8 and graph_error0; resident service launches1 and host264. Every process has zero registration/admission drops, all eight final slots FREE, and module/hook/pin cleanup. Approximate cost is ten bank-capture starts plus the complete serving matrix, around fifteen minutes on the shared device; it is not a fixed investigation limit.

Preserve stdout and raw request/cell/bootstrap/idle/executor/failure records. A partial prefix or successful trigger is incomplete. Keep failed attempts with distinct paths and unchanged oracle. Fresh plan review precedes real preflight; a fresh result reviewer independently reconstructs counts, timestamps, metric signs, CPU counter scope, resource matching and all reported effects. Review is diagnostic, not an extra permission gate. The intended table contains both matched modes and every paired effect; historical serial-control results remain unchanged.

## Correctness result

Both native Makefile targets build successfully after the shared dispatcher change. The [concurrent host raw](../../bench/results/qwen_concurrent_host_control_correctness_20261011.txt) and [resident raw](../../bench/results/qwen_concurrent_resident_correctness_20261011.txt) each complete the unchanged strict regression:335 valid requests/20432 exact generated tokens,259 rejects,5000 empty connections, eight persistent clients, mixed decode lengths, immediate reuse, wrap, ordered pipeline and half-close. Both record20432 model graphs,graph_error0,banks8,max_active8 and eight final FREE slots; service launches are594/1 respectively. Independent raw reconstruction checks all39 printed expected/actual arrays and the after-ready network syscall search; the full original traces are preserved in those raws. Resident qualification has an explicit exit0 receipt. The first host launcher did not retain a wait()/exit-code receipt; its terminal PASS output and absent process are recorded without inventing that status.

These qualify the concurrent control and re-entry/lifetime behavior, not a serving speedup. Old one-bank results remain historical. No arithmetic or upstream-backend change follows.


## Independent plan review

A fresh read-only reviewer checked admission, the native phase1/phase3 state
transition, FIFO retirement and host re-entry, identical eight-bank capture,
strict correctness receipts, published metric definitions, paired sampling,
commands, resource conditions and interpretation. The review found no scientific
or execution defect that invalidates the planned component comparison, so it
can proceed to real preflight and full collection.

The review identified missing paired TPS intervals and completion aggregation
in the existing analysis output. The minimal analysis change now reports
resident-minus-host TPS for every pair and its nominal df4 t95 interval,
completion mean and the descriptive whole-host CPU observations. One focused
read-only follow-up verified the signs, formula and diagnostic scope. Preflight
has one pair and reports null intervals; it cannot establish a performance
effect.

Fresh result review must independently check the complete unique matrix,
request indices and prompt balance, exact tokens, timestamp-derived metrics,
native launch counts, zero drops and final FREE slots. These are evidence checks,
not a new runtime control interface. No reviewer ran GPU work or edited source.


## Real preflight

The [raw preflight](../../bench/results/concurrent_graph_dispatch_preflight_20261011.jsonl)
and [command/stdout receipt](../../bench/results/concurrent_graph_dispatch_preflight_20261011.stdout.txt)
complete with exit0 at2026-10-11T05:29:01.658986UTC. Both modes use the actual
default TCP path, eight graph banks and unchanged gen64 oracle warmups.
Independent reconstruction confirms two unique measured cells, four prompt
cases/mode, eight measured exact gen1 tokens,24 total request publications
including warmups and528 model graphs. Resident/host service launches are1/12;
both have graph_error0, zero registration/admission drops and eight final
FREE slots. No module or bpf-gpu-server map pins remain after cleanup.

This confirms executability only. One paired preflight block produces null
confidence intervals and is not evidence of a serving speedup.


## Full collection and reconstruction

The [complete raw](../../bench/results/concurrent_graph_dispatch_20261011.jsonl),
[command/stdout](../../bench/results/concurrent_graph_dispatch_20261011.stdout.txt)
and [independent reconstruction source/output](../../bench/results/concurrent_graph_dispatch_20261011.analysis.txt)
record collection on revision870c9d18a76015879167e3acaf47767bdf18ab12,
2026-10-11T05:30:54.592270 through05:44:58.678897UTC, exit0.
An initial SSH banner timeout did not launch a process or create a raw/stdout
file; authoritative inspection confirmed their absence before the same command
was retried. No in-experiment model, source, oracle, timeout or TCP-setting
deviation occurred.

Independent reconstruction verifies40 unique planned cells,2560 measured
requests,83200 exact measured tokens, five complete pairs and ten normal
executor exits. Every cell contains indices0..63 once and16 requests per
prompt. Mode order alternates, and each pair retains the same rotated cell
order. Every process publishes264 requests and executes8580 model graphs with
banks8,max_active8,graph_error0; resident/host service launches are1/264.
All process receipts show zero registration/admission drops, zero parser
invalids and eight final FREE slots.

| Generated tokens | Clients | Mode | Mean TTFT ms | Mean TPOT ms | Mean completion ms | Mean cell output tok/s | Mean executor CPU s/cell |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1 | resident | 40.932 | — | 40.932 | 24.409 | 0.002 |
| 1 | 1 | host-launch | 41.583 | — | 41.583 | 24.025 | 2.664 |
| 64 | 1 | resident | 40.748 | 2.898 | 223.312 | 286.542 | 0.006 |
| 64 | 1 | host-launch | 42.285 | 2.883 | 223.905 | 285.780 | 14.344 |
| 1 | 8 | resident | 39.500 | — | 39.500 | 193.346 | 0.000 |
| 1 | 8 | host-launch | 40.795 | — | 40.795 | 190.423 | 0.340 |
| 64 | 8 | resident | 46.386 | 11.033 | 741.450 | 688.143 | 0.006 |
| 64 | 8 | host-launch | 45.215 | 11.139 | 746.973 | 684.580 | 5.990 |

TTFT/TPOT/completion summarize320 measured requests per row. Output TPS is the
mean of five cell-level aggregate TPS measurements, not per-request throughput.
CPU seconds cover64-request cells; zero readings are subject to the100Hz CPU
counter resolution. Startup means are54.177s resident and53.824s host, with
ranges52.850-57.232/52.900-54.381s. Each one-second idle interval records0/1.0
executor CPU seconds respectively.

| Tokens / clients | Host-minus-resident TTFT ms, nominal t95 | Resident-minus-host output tok/s, nominal t95 |
|---|---:|---:|
| 1 / 1 | +0.651 [0.341,0.962] | +0.383 [0.204,0.563] |
| 64 / 1 | +1.537 [1.352,1.723] | +0.761 [-1.474,2.997] |
| 1 / 8 | +1.295 [0.748,1.842] | +2.923 [-1.333,7.179] |
| 64 / 8 | -1.171 [-1.632,-0.711] | +3.563 [-4.800,11.925] |

The full five effects for every cell are preserved in stdout and reconstruction;
intervals use paired process-block means and df4. Shared device conditions,
adjacent-run correlation and a small sample limit inference; these are nominal
local intervals, not a family-wide or robust-tail claim.

The co-primary concurrent gen64 condition has a negative TTFT effect: keeping
service dispatch resident is1.171ms slower here, about2.6% of host-control TTFT.
Its TPS interval crosses zero, so this collection does not establish a
throughput improvement or equivalence. Other cells have small positive TTFT
effects; only the gen1/single-client TPS interval excludes zero. Completion
and TPOT differences are descriptive diagnostics.

The resident policy removes most process CPU work relative to this synchronous
busy-polling request-retirement control. It does not establish that an optimized
asynchronous userspace server would consume a core or lose throughput. Both
modes already retain eBPF ingress and kernel TX; this component ablation does
not measure replacing a conventional userspace network request worker.
Whole-host CPU seconds include clients and unrelated work, so they remain
descriptive and cannot independently establish attributed service or energy
savings.

Before/after live host observations, copied from the root-thread command outputs
into the stdout receipt (not generated by the benchmark driver), retain the
same boot and unrelated SGLang
PID2762187 at18346MiB. Matching kernel-journal Xid searches have no entries;
after collection the bpf-gpu-server module and map pins are absent. No unrelated
GPU process, service or persistent setting was changed. This comparison uses
loopback/default TCP and is not an isolated GPU or physical-NIC result.

The result bounds dispatch as the next optimization target: it provides no
basis for another unchanged dispatch repetition or a broad latency/throughput
win. The higher-value continuation is to qualify and integrate upstream
computation with minimal request/state/lifetime glue, then compare the same
backend through credible userspace and resident/kernel request paths. Model
arithmetic tuning remains a separate claim.


## Independent result review

A fresh read-only reviewer independently read the plan, collector and native
source, reconstructed the raw matrix, exact reference tokens, request timestamps,
paired effects and CPU counters, then checked every reported table value and
README rounding. The full run is valid. The concurrent gen64 TTFT improvement
prediction is contradicted; its output TPS improvement is inconclusive.
Research value is supporting, with a mechanism/workload boundary rather than
a direct thesis result. The reviewer did not run GPU work or edit source.

Ten executor events follow checked normal exits. Native counts, zero drops,
eight FREE slots and cleanup receipts agree. One host pair retains a
BF_WRITING publication reservation diagnostic; source retries it, and that
process has zero drops and all exact completions, so no row is excluded.
The review originally lacked the separately observed boot/SGLang/Xid receipts;
these are now transparently appended to stdout as root-thread observations,
without presenting them as a benchmark-driver or reviewer reconstruction.

The review recommends ending unchanged dispatch repetitions and directing
the main comparison toward a qualified upstream compute backend with credible
userspace and resident/kernel request paths. It supports neither general
latency/throughput superiority nor attributed whole-service CPU savings.
