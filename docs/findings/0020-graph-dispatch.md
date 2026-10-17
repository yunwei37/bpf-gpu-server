# 0020: matched whole-request dispatch with device-scheduled Qwen graphs

## Engineering question and admission

The retained objective requires reproducible correctness and matched controls
for Linux TCP/HTTP -> eBPF/shared queue -> resident real-model inference ->
GPU completion -> kernel TCP TX, without a steady userspace request worker.
There is no paper RQ, thesis or evaluation contract in this repository. This
supporting mechanism experiment does not invent one or complete the service goal.

Question: does GPU-resident whole-request dispatch improve client-observed
TTFT when both policies use the now-correct, device-tail HF computation?
The prediction is a positive host-minus-resident mean TTFT difference in at
least some tested cells. A negative interval contradicts that prediction for
its cell; crossing zero is inconclusive, not equivalence. CPU is secondary.
This tests an uncertain benefit, not the definitional absence of a host worker.

Finding 0018 already bounds the older scalar computation and one/eight-token
workload. It cannot be reanalyzed into graph/64-token results: model execution,
GPU idle policy and the runnable long-output scope changed. This adds a
matched integrated measurement of the new real path. Positive results support
only this local policy benefit; null/negative results bound it and end this
ablation. Neither would establish competitive serving or a paper-thesis verdict.

The strongest next end-to-end alternative is an optimized official serving
comparison with complete host accounting and physical-NIC delivery. It remains
more consequential, but cannot isolate this policy effect with the changed
math engine. Reuse the existing native protocol benchmark and exact-HF control
now, then prioritize those broader requirements; do not repeatedly rerun an
unchanged null ablation. This experiment is supporting, not headline evidence.

## Comparison and assets

The proposed path is `build/qwen_graph`; the component ablation is
`build/qwen_graph_host_launch`. There is no main external baseline here.
Both use the same capture script, original exported Qwen2.5-0.5B fp16 weights,
HF eager operations, graph catalogue, KV, device argmax, queue, parser, kernel
accept/TX and dispatcher instructions. A uniform `once` argument changes
whole-request return: one host service launch versus one per whole request.
Both retain GPU-tail decode. This does not isolate per-token graph-launch cost
or compare GPU-tail dispatch against an official optimized host server.

Current assets: PyTorch 2.14.1+cu130, Transformers 5.19.0, CUDA 13.0.88,
Linux 7.3-rc3, one RTX 5090 in the original bpfusion-dev Workspace. Finding
0019's two-mode gen1/gen64 regressions establish setup correctness, not speed.
No oracle, model weights, timeout or precision change is permitted during a run.

Default TCP buffering previously coalesced token arrivals. For this bounded
experiment only, both native libraries are built with a checked listener
TCP_NODELAY call. Restore tracked source immediately after native make;
preserve default binaries. No transport setting is retained. Compare candidate
and default device SASS/resources and run both complete strict regressions
on the candidates before preflight. This is a shared transport condition,
not evidence of a transport-only causal effect against older sequential runs.

## Workload, metrics and interpretation

- Four existing real prompts with independently computed eager-HF fp16 greedy
  oracle sequences, loaded/released before serving.
- Output lengths 1 and 64, closed-loop client concurrency 1 and 8. Requests
  are serial on GPU; client concurrency is queueing, not continuous batching.
- Five paired process repetitions, alternating mode order; identical rotating
  cell order within each pair. Sixty-four requests per cell (16 per prompt).
- Eight excluded warmups per process: each prompt at gen1 and gen64. Model
  load/capture, warmup, connection/thread creation, GPU queries and teardown
  are outside the measurement and process-CPU windows. Idle measured separately.
- [NVIDIA latency conventions](https://docs.nvidia.com/nim/benchmarking/llm/metrics):
  TTFT from client send start to first complete token; TPOT is (last-first)/63
  for gen64 and null for gen1. Completion latency and output-token throughput
  are secondary. One recv can timestamp multiple tokens equally. These are
  client arrivals including transport and kernel polling, not GPU-ready times.
- Primary effect per gen/concurrency: paired host-minus-resident mean TTFT
  over five process blocks, two-sided 95% Student-t interval, df4. The
  [NIST t table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm)
  supplies the critical value. Assumptions include approximately independent,
  normally distributed block differences; shared hardware and five blocks
  limit inference. Cells are separate; no family-wide superiority claim.
- Pooled p50/p99 (320 requests per cell/mode) are descriptive, not robust tails.
- Executor-process utime+stime at recorded tick resolution includes its threads,
  excludes kernel TX/softirq/clients/other host work, and is not idle-subtracted.
  Quantized near-zero values are not exact zero or savings-ratio denominators.
- GPU clocks/temperature/power are endpoint observations. Policy combines
  polling, host launch/sync, re-entry and power behavior; no isolated launch
  cost or integrated energy claim. Captures/startup/memory are not timed here.
- Strict tokens, mechanism engagement, matched resources and complete cells
  are validity conditions. Secondary CPU is not a veto of the latency hypothesis.

## Execution and completion

Use `make qwen-graph qwen-graph-control` for default code. To build the
bounded candidates, temporarily add `#include <netinet/tcp.h>` and after
listener SO_REUSEADDR add a checked
`setsockopt(listener,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes))`. Native
`make BUILD=/tmp/bpfusion-graph-dispatch-nodelay-... qwen-graph qwen-graph-control`
creates libraries/aliases together. Restore source before executing. Record
exact source delta, native build, candidate path and SASS/resource inspection.
The alias loads its adjacent library, so no copied capture script is needed.

With `candidate` set to that output directory:

```sh
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor "$candidate/qwen_graph"
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor "$candidate/qwen_graph_host_launch"
HF_HOME=/workspaces/.cache/huggingface python3 bench/resident_dispatch.py --gen 64 --preflight --resident-executor "$candidate/qwen_graph" --host-executor "$candidate/qwen_graph_host_launch" --output bench/results/graph_dispatch_preflight_20261010.jsonl
HF_HOME=/workspaces/.cache/huggingface python3 bench/resident_dispatch.py --gen 64 --resident-executor "$candidate/qwen_graph" --host-executor "$candidate/qwen_graph_host_launch" --output bench/results/graph_dispatch_20261010.jsonl
python3 bench/resident_dispatch.py --analyze --output bench/results/graph_dispatch_20261010.jsonl
```

Preflight uses four exact gen1 requests per mode after gen1/gen64 warmups.
A complete full run contains all five pairs, ten normal process shutdowns,
forty unique cells, 2560 measured requests and 83200 exact output tokens.
Each process serves eight warmups plus 256 requests: host service launches
1/264 and device model graph launches 8580 in either mode, graph_error=0.
Every cell has indexes0..63 exactly once. Parser publications are 264/process,
with zero registration failures/drops and clean final slots/teardown.

The existing driver writes each completed cell and preserves partial request
rows plus failure-time queue/log statistics. Preserve failed/incomplete attempts
with distinct raw paths; never overwrite them or count prefixes as complete.
Capture stdout separately. Independent fresh plan review precedes preflight;
a different fresh result reviewer reconstructs metrics/outputs and engagement.
Temporary binaries remain for result inspection, then owner removes them after
all experiment processes exit. The target output is one per-cell mean/effect
TTFT table with intervals plus scoped secondary CPU/TPOT observations.

## Plan-review boundary correction

Review found that the old runner started its process-CPU/window timer before
workers finished creating connections. TTFT starts at each send and throughput
uses first-send/last-arrival, so those metrics were unaffected; CPU/window
could include residual setup. A three-line ready-barrier handshake now waits
for every connected worker before starting those windows and releases workers
through the original timed barrier. Oracle, timeouts, cell order, counts and
metric formulas are unchanged. This is corrected before preflight/full run;
older findings retain their historical measurement code and scope.

## Candidate admission failure and repair

The first TCP_NODELAY candidate regression timed out while eight persistent
clients alternated invalid and valid requests. The retained
[failed log](../../bench/results/graph_dispatch_candidate_resident_correctness_20261010.txt)
reports one busy admission/drop and zero graph errors. Its free-slot sample
was collected after the timeout; it does not identify the state at refusal or
prove the historical cause.

Temporary refusal tracing did not reproduce that drop. The
[gen64 instrumented regression](../../bench/results/graph_dispatch_candidate_resident_diagnostic_20261010.txt),
[fast resident stress](../../bench/results/graph_dispatch_fast_resident_diagnostic_20261010.txt),
and [fast host stress](../../bench/results/graph_dispatch_fast_host_launch_diagnostic_20261010.txt)
passed; the associated
[initial trace](../../bench/results/graph_dispatch_reservation_trace_20261010.txt)
and [stress trace](../../bench/results/graph_dispatch_fast_reservation_trace_20261010.txt)
are empty. This is absence of a captured refusal, not causal proof. Temporary
trace calls and copied stress scripts were removed. The
[gen8 diagnostic stress](../../bench/results/graph_dispatch_reservation_word_diagnostic_20261010.txt)
also passed 2,095 valid requests / 16,760 exact tokens and 2,051 rejections.

Source inspection found that a producer could see a stale head or another
producer's WRITING reservation and immediately classify it as a full ring.
The existing bounded reservation loop now retries those publication-contention
cases. Successful publication still requires both slot and head CAS; a failed
head CAS releases that producer's reservation. This fixes the source-level
premature-full behavior, but does not promise lossless admission under true
overload or prolonged contention, and does not establish the old drop's cause.

One aligned word in the previously unused LLM padding records the last busy
reservation's low 29 head bits, observed state and whether head changed.
The loader prints it when nonzero. It is best-effort, truncated, concurrently
overwriteable observation; it may describe recovered contention rather than
a dropped request. There is no new map, queue size, timeout or trace call.

After the retry repair, both complete candidate gen64 regressions passed the
unchanged oracle: **303 valid requests / 19,392 exact tokens and 259 rejections
per mode**, no admission drops, graph_error=0 and clean module/pin teardown.
[Resident log](../../bench/results/graph_dispatch_reservation_retry_resident_correctness_20261010.txt)
records one host service launch; [host-control log](../../bench/results/graph_dispatch_reservation_retry_host_launch_correctness_20261010.txt)
records 562. Both record 19,392 device model graph launches and all-thread
network-syscall checks. These are correctness results, not speed measurements.

Fresh plan review and its focused follow-up admitted the supporting comparison
and found no new source blocker after the retry and timer-boundary correction.
The review discussions remain in the retained task: automatic approval rejected
two reviewer report-save operations with only "blocked by policy", so no saved
review report is claimed.

## Complete matched result

The restored-runtime full run completed once, at source commit `e3f272e`,
with **40 cells, 10 normal executor shutdowns, 2,560 measured requests and
83,200 exact output tokens**, without failure events. All ten processes had
8,580 model graph launches, graph_error=0, 264 publications, zero admission
or registration failures and eight final FREE slots. The
[full raw](../../bench/results/graph_dispatch_20261010.jsonl),
[stdout and analysis](../../bench/results/graph_dispatch_20261010.stdout.txt),
[real preflight](../../bench/results/graph_dispatch_preflight_20261010.jsonl),
[preflight stdout](../../bench/results/graph_dispatch_preflight_20261010.stdout.txt),
[rebuilt native candidates](../../bench/results/graph_dispatch_rebuilt_candidate_20261010.txt),
and rebuilt strict
[resident](../../bench/results/graph_dispatch_rebuilt_resident_correctness_20261010.txt) /
[host](../../bench/results/graph_dispatch_rebuilt_host_launch_correctness_20261010.txt)
regressions preserve the evidence. Both rebuilt strict regressions again
passed 303 requests / 19,392 exact tokens / 259 rejections. The source, oracle,
workload and deadlines were unchanged during measurement.

The table reports pooled request means; uncertainty is computed from the
five paired process-block mean differences, host minus resident. Each cell
has 320 measured requests per mode.

| output / clients | resident TTFT ms | host TTFT ms | host minus resident ms | paired t95 ms |
|---|---:|---:|---:|---|
| 1 / 1 | 2.739621 | 2.913690 | +0.174069 | [-0.100007, +0.448145] |
| 64 / 1 | 2.797782 | 2.806058 | +0.008276 | [-0.083723, +0.100274] |
| 1 / 8 | 20.276981 | 20.334831 | +0.057851 | [+0.037308, +0.078393] |
| 64 / 8 | 991.274073 | 972.232360 | -19.041714 | [-56.727901, +18.644474] |

Only the one-token/eight-client cell supports a small resident TTFT benefit.
The other three cells are inconclusive. Retain the negative long-output point
estimate: two resident blocks were slower; neither their cause nor a general
resident disadvantage is established. Crossing zero does not prove equivalence.
There is no broad or family-wide latency-superiority conclusion.

| output / clients | resident / host TPOT ms | resident / host output tokens/s |
|---|---|---|
| 1 / 1 | not applicable | 363.118 / 343.060 |
| 64 / 1 | 2.311563 / 2.324503 | 431.143 / 428.793 |
| 1 / 8 | not applicable | 373.397 / 372.477 |
| 64 / 8 | 2.345387 / 2.302970 | 425.800 / 433.239 |

TPOT is the pooled request mean; throughput is the arithmetic mean of the
five measured cell rates, not a single merged-window rate. These secondary
observations are descriptive. Eight clients do not increase throughput through
continuous batching: GPU requests remain serial, and long-output TTFT grows
to about one second. This identifies queueing as a consequential next target.

Resident executor CPU samples were 0 or 0.01 seconds per cell; host-control
samples were 0.17-0.21 seconds for gen1 and 9.42-9.78 seconds for gen64.
Each host idle sample consumed about one CPU second in one wall second.
Resident values are quantized at 100 Hz, not exact zero. The control's concrete
poll/synchronize implementation accounts for this cost; optimized host serving
need not share it. Kernel TX/softirq/client/other-process cost and integrated
energy remain unmeasured. GPU endpoint readings cannot explain within-window
variation or attribute the two slower resident blocks.

A fresh independent result reviewer read source/build/oracle/complete raw and
recomputed every request's TTFT, TPOT and completion latency, every cell's
throughput, descriptive quantiles and paired intervals. Completion, exact
outputs, matching dispatcher instructions/transport, exclusions and cleanup
passed. The review returned at 2026-10-10 18:15 UTC: run **valid**, hypothesis
**supported narrowly** in 1/8 and otherwise inconclusive, value **supporting**,
impact **mechanism/workload boundary**. There is no selected paper RQ here.
The decision is to end this ablation and move to computation/queueing and
stronger serving evidence, rather than rerun it unchanged. The reviewer made
no file writes or GPU runs; this paragraph records its returned review.

## Status

Plan, real preflight, complete matched run and fresh result review finished.
The local dispatch ablation is closed; the full service goal remains active. On 2026-10-10 at 17:55 UTC, live inspection
found the earlier container gone and the same Workspace's existing replacement
Pod running, with the original PVC/source/logs retained and the same host boot
ID. Both completed correctness logs survived. Disposable candidate binaries
and container-installed dependencies did not. Build dependencies were restored
with the default package path in this original Workspace; PyTorch 2.14.1 and
Transformers 5.19.0 were restored to preserve the retained comparison. Candidates
were rebuilt and both strict regressions repeated before measurement. The
original valid kernel-module binary survived; no fresh module rebuild is claimed.
Container termination cause
has not been established by this inspection.
Full service scope remains text processing, continuous batching, completion
interfaces/timing, complete host cost, optimized serving and physical NIC.

## Official token-only optimized baseline dependency preparation

After the complete paired-KV collection and its after-host snapshot,the next
comparison advances to official optimized serving. This is benchmark-only
dependency setup,not deployment,inference preflight or a performance result.

The existing bench/run_http.sh lifecycle uses broad pkill and its
tools/llm_http_client.py counts nonempty SSE text chunks as tokens while
adding chat formatting. It cannot be reused unchanged for a shared-host
matched token-level comparison. Official vLLM0.31.0 supports
/inference/v1/generate with --tokens-only:token IDs in/out,no tokenizer or
detokenization. Its installed serving source uses DELTA output for streams.
Use actual choice.token_ids and their arrival times,not SSE chunk count.
The versioned source lacks the newer output_mode field shown in latest docs;
use the installed/versioned API,not unsupported latest parameters.

Original bpfusion-dev Workspace/lab RTX5090/PVC/session/other SGLang owner
remain preserved. Both inspected base environment and existing .venv-ml lack
vLLM;the latter also lacks Torch/Transformers. An OS-temp environment
/tmp/bpfusion-vllm-uapngodu/venv holds official vLLM dependencies,leaving
qualified Torch2.14.1/Transformers5.19.0/Triton3.8.0 unchanged.
A first python3 -m venv attempt fails before package/model activity because
ensurepip is absent. Existing pip24.0 supports --python targeting the already
created temporary venv;no global apt package/original environment change.

Successful command:
python3 -m pip --python /tmp/bpfusion-vllm-uapngodu/venv install vllm==0.31.0 --extra-index-url https://download.pytorch.org/whl/cu129

[Full install log](../../bench/results/qwen_vllm_dependency_install_20261010.txt)
terminates0. [Version/import/dependency receipt](../../bench/results/qwen_vllm_dependency_versions_20261010.txt)
records vLLM0.31.0,Torch2.13.0+cu129,Transformers5.17.0,
FlashInfer0.7.0.post1,httpx0.28.1. CPU vLLM/Torch import and pip check pass.
This does not prove compatible CUDA execution,model readiness,correct
generation or performance. No model/server/GPU inference run has occurred.

The cached upstream Qwen2.5-0.5B model revision
7ae557604adf67be50417f59c2c2f167def9a775 is available for the next frozen
comparison. Root owns this temporary environment for only that benchmark;
after terminal benchmark and frozen reproduction artifacts,confirm no live
process uses it and remove this exact resolved /tmp directory. Future serving
must remain private loopback,with exact owned-process lifecycle/authentication,
no broad pkill,new Workspace,gateway,Secret,service or persistent setting.
Original model/oracle/native/test source and unrelated tools/setup_env.sh
remain unchanged. Plan/independent review and real preflight precede any
competitive performance claim;baseline execution or numerical failure is
not a bpf-gpu-server win.

Sources:[vLLM0.31 token client](https://github.com/vllm-project/vllm/blob/v0.31.0/examples/scale_out/token_generation_client.py),
[versioned serving source](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/scale_out/token_in_token_out/serving.py),
[official installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/),
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/).
