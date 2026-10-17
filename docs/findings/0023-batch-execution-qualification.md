# 0023: batch execution qualification

## Why qualify computation before implementing batching

Finding0022 removes repeated KV prefix work but leaves serial requests and large eight-client queueing. The next service work is concurrent/batched execution without a steady host request worker, with independent KV ownership and the same exact eager-fp16 oracle. These are engineering qualification probes, not standalone research or serving-speed results.

## Ordinary HF batch-shape candidate

The [complete raw](../../bench/results/qwen_batch_shape_qualification_20261010.txt) includes the exact temporary script, source commit97a5f74, installed versions, independent batch1 eager-HF oracle sequences, command and all actual outputs. It uses the original exported weights through load_model, eager attention, fp16, last-position logits and ordinary HF DynamicCache. It does not run TCP, the native dispatcher or a resident service. Its host-driven calls are a diagnostic, not the final worker-free path.

The full matrix uses batch2/4/8, each of the ten gen64 correctness/prompt-edge cases replicated across lanes, plus a same-prefix-length mixed batch of the real France/greeting prompts. Every lane owns its own KV state in the HF batch. All33 groups completed and produced9,856 tokens;26 groups pass and7 fail exact oracle agreement. Exit0 means the diagnostic matrix completed, **not** that the candidate passed. No group was omitted after a mismatch.

| batch | failing group | first divergent zero-based output | expected / actual |
|---|---|---:|---:|
| 2 | greeting (case2), mixed5 greeting lanes | 62 | 323 /11 |
| 4 | arithmetic (case1) | 20 | 5145 /64547 |
| 4 | one-token prompt edge (case4) | 31 | 36277 /7010 |
| 8 | greeting (case2), mixed5 greeting lanes | 62 | 323 /11 |
| 8 | one-token prompt edge (case4) | 31 | 36277 /7010 |

The tested batch-shape change diverges under the fixed strict oracle. This does not establish that ordinary HF batching is generally wrong or isolate which arithmetic/dispatch difference causes divergence. The oracle and precision standard are unchanged; no tensor-batched production change is retained, and no latency/throughput figure is reported from this probe.

## Independent batch1 graph-bank qualification

The alternative keeps each request's qualified batch1 calculation shapes and private input/KV/output tensors while sharing read-only model weights. A GPU dispatcher launches each bank's selected model graph with CUDA FireAndForget, then tail-launches itself to sample outputs and feed the next decode step. CUDA's [device graph execution environments](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) include child work: the parent tail waits for the whole environment, including the fire-and-forget children. This is a GPU-controlled fork/join route, not evidence that model kernels actually overlap.

The [temporary native build raw](../../bench/results/qwen_graph_bank_build_20261010.txt) preserves the exact addon, Makefile command and output at source2062f44. The addon was appended only for its temporary build; production native source was restored unchanged. Each raw below contains the exact wrapper and independent batch1 oracle outputs. Captured graphs and all tensor keepers remain alive through native graph-exec destruction. The first [bank2 attempt](../../bench/results/qwen_graph_bank2_qualification_20261010.txt) completed its first128 model graphs with zero GPU error, then failed while printing because json was absent from exec globals. That failed attempt is retained and is not a completed qualification. The repair adds only json to those globals; capture, scheduler, oracle and evaluation are unchanged.

| private banks | groups passing / completed | exact output tokens | model graphs per job | host launches per job |
|---:|---:|---:|---:|---:|
| [2](../../bench/results/qwen_graph_bank2_qualified_20261010.txt) | 12 /12 | 1,536 | 128 | 1 |
| [4](../../bench/results/qwen_graph_bank4_qualified_20261010.txt) | 12 /12 | 3,072 | 256 | 1 |
| [8](../../bench/results/qwen_graph_bank8_qualified_20261010.txt) | 12 /12 | 6,144 | 512 | 1 |

Each bank-count matrix includes all ten gen64 correctness/prompt-edge cases replicated across banks, a mixed France/greeting group, and a mixed-prompt-length group. All36 groups complete with exact agreement on10,752 tokens; all36 native markers report error=0 and one host launch per whole job, with no per-decode host model calls. These are engineering qualifications, not serving benchmarks: there is no TCP/shared BPF queue/module execution, latency measurement or demonstrated kernel overlap. At this GPU-only stage the production service remained serial.

## Real TCP private-bank integration

The default resident graph executor now discovers bank count from the existing BF_LLM_SLOTS authority and captures a private input/KV/output graph bank per physical request slot. Model weights remain shared and each model computation is still batch1 eager fp16. The GPU admits published requests in ring order, advances active requests through FireAndForget graphs, and samples only after the parent tail joins all child work. Requests can complete computation out of order. DONE publication follows admission order and waits until kernel TX has released the previous DONE before publishing the next one; FREE or already reused WRITING/PENDING proves that release. This preserves the existing producer's head-indexed reservation without a page ABI, producer or module change. The module continues withholding the final token while the GPU owns PENDING. Private banks remain alive through native graph-exec destruction.

The [resident real TCP strict raw](../../bench/results/qwen_queue_banks_resident_mixed_correctness_20261010.txt) passes335 valid requests /20,432 exact tokens /259 rejects,5,000 empty connections, eight clients, invalid-to-valid reuse, persistent ring wrap, a new32-request mixed gen1/gen64 reuse check, ordered pipeline and half-close. It reports one host launch,20,432 device model graphs,error=0,banks=8,max_active=8,zero admission/registration drops and all slots FREE at shutdown. The all-thread network trace contains no accept/receive/send calls after ready. This qualifies actual queue/TCP ownership and computation, not kernel overlap or serving speed.

The initial [startup-only run](../../bench/results/qwen_queue_banks_resident_correctness_20261010.txt) exceeded the inherited sixty-second readiness wait while capturing eight banks. No requests were published and no module loaded; it is retained as a failed setup attempt. Readiness now follows ready or process exit; the independent oracle and request/shutdown timeouts are unchanged. More capture work and memory are real costs, excluded from steady-state timings and reported separately where measured.

The serial host-launch correctness control captures one bank and returns after one whole request. Its first [mixed strict run](../../bench/results/qwen_queue_banks_host_mixed_correctness_20261010.txt) failed: each parent-tail iteration admitted another queued request while the current one was unfinished, so two physical slots shared bank0 (max_active=2,graph_error=1) and clients timed out. The sole correction requires seen==retired before admission when once=true. This does not change the resident once=false path. The [corrected host raw](../../bench/results/qwen_queue_banks_host_mixed_corrected_20261010.txt) passes the same335 requests /20,432 exact tokens /259 rejects and all lifecycle/trace checks, with594 host launches,20,432 model graphs,error=0,banks=1,max_active=1,zero admission/registration drops and eight FREE slots. The failed raw remains a failure. Resident qualification predates this once-only guard; the reviewed false branch has unchanged behavior.

This implementation permits concurrent request progression with separate batch1 calculations. At this correctness milestone, tensor batching, measured simultaneous kernel overlap, serving benefit, text processing and physical-NIC coverage remained separate open requirements. The completed component comparison below now measures serving benefit in its specified shared-GPU setting.



## Real TCP concurrency comparison plan

There is no paper RQ/thesis in this repository. This is a supporting engineering comparison for the user's throughput/latency objective: does admitting and progressing independent batch1 graph banks reduce eight-client gen64 queueing/TTFT and increase total output-token throughput against the current serial resident service? Finding0022 already establishes serial queueing; GPU-only qualification cannot answer the real-client question. The strongest next comparison changes that serving mechanism. Repeating dispatch or KV-only ablations would not answer it.

Baseline is the serial resident service at2062f44 (same code as10ad405, whose change is records only), including the accepted shared-KV append and last-position logits. Candidate is the qualified real TCP bank executor: private inputs/KV/outputs for the existing eight ring slots, GPU FireAndForget children followed by parent tail sampling, FIFO DONE publication with previous kernel reclamation acknowledged before the next DONE. Producer, page ABI, kernel module, weights, eager fp16 math, strict independent batch1 oracle and request transport remain unchanged. Both engines use one host service launch. The serial engine is a current-practice/component comparator, not an external SOTA baseline. The serial host-launch control is correctness coverage, not part of this concurrency comparison.

Use the existing bench/resident_dispatch.py protocol through its previous shared-KV temporary adapter, renamed serial/banks. Both temporary native libraries enable the same checked listener TCP_NODELAY call used in finding0022; production default transport is unchanged. Record exact sources, native build command and adapter delta. The bank startup exceeded the old sixty-second capture wait before requests; readiness now follows ready or process exit, with capture excluded from measurement and original request/shutdown timeouts retained. Record that deviation before collection.

Matrix: four real prompts, gen1/gen64, clients1/8, five paired process blocks with alternating engine order and rotating matching cell order,64 requests per cell (16 per prompt), eight excluded warmups per process, one-second idle samples separate from active cells. Full collection is40 cells,2,560 requests,83,200 measured output tokens and ten independent process starts/shutdowns. Preflight reuses the driver's gen1/client1/four-request cell for each engine; it establishes executability only. Connection/thread/barrier creation, model loading and capture precede timing/CPU windows. Record bootstrap duration separately for its larger bank-capture cost.

[NVIDIA metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics) define primary gen64/clients8 total output TPS over first-send/last-arrival, and request TTFT from send to first complete token. Report five paired block differences and nominal two-sided t95 intervals (df4): banks-minus-serial for TPS, serial-minus-banks for latency. Co-primary effects are assessed separately; a positive one with an inconclusive other is mixed support, while a clear tradeoff is reported as such. TPOT=(last-first)/63 and completion latency, single-client effects, gen1, executor-process CPU and idle CPU remain reported diagnostics/tradeoffs, not silent vetoes. Multiple tokens completed in one recv share its arrival timestamp; no weighted score or familywise correction.

Positive eight-client effects support retaining this serving path; negative/inconclusive effects bound its benefit and direct the next implementation toward measured compute/queueing overhead. All requests must match the independent oracle, counts must match, both service launch counts must be one, graph/admission errors must be zero, final slots must be FREE and test hooks/module/pins must clean up. The native max_active marker can prove multiple graph requests in flight, not simultaneous kernel overlap. Retain every paired block and any failed run. No compiler/precision/oracle changes or post-hoc sample removal.

Shared GPU: another project's SGLang workload was observed in finding0022. Preserve it. Record GPU clocks/temperature/power/utilization before/after each cell and current host process residency before/after the run through existing diagnosis paths. These are coarse observations, not an activity trace or isolation proof. Alternating pairs limit but do not eliminate temporal confounding; inference is local to this shared-GPU run, with no fixed isolated speedup, energy, open-load capacity, physical-NIC or optimized-serving claim.

Command after qualification/build/plan review: HF_HOME=/workspaces/.cache/huggingface python3 /tmp/<owned-queue-bank-directory>/queue_banks_bench.py --gen 64 --output bench/results/queue_banks_20261010.jsonl --resident-executor /tmp/<owned-queue-bank-directory>/serial/qwen_graph --host-executor /tmp/<owned-queue-bank-directory>/banks/qwen_graph. Add --preflight with its own output path for setup. The exact instantiated command, paths and script are retained in raw artifacts. Final table reports both primary effects with all paired blocks plus the TPOT/single-client tradeoffs. Plan and result reviews are read-only chat verdicts recorded here; no separate saved report is claimed.

Fresh read-only plan review admitted this comparison on the corrected source and exact temporary build/adapter. It verified the new decision value, matched serial comparator, metric signs/counts, private-bank lifetimes, FIFO reclamation and small production growth. The failed host run and applied once-mode guard were explicitly reviewed; finish its qualification and real preflight before full collection. The inherited primary_metric output label was renamed metric before collection; interpretation follows the declared TPS/TTFT co-primary plan, with TPOT diagnostic. Completion latency must be aggregated from stored request timestamps in the final results. No separate saved review report is claimed.

Real preflight completed on source41e5fc4: [raw](../../bench/results/queue_banks_preflight_ready_20261010.jsonl) /[stdout](../../bench/results/queue_banks_preflight_ready_20261010.stdout.txt),15 records/eight exact measured gen1 requests, two normal process shutdowns, each12 publications/264 model graphs/one host launch/error0/zero drops/eight FREE slots and cleanup. The [first preflight](../../bench/results/queue_banks_preflight_20261010.jsonl) /[stdout](../../bench/results/queue_banks_preflight_20261010.stdout.txt) failed at Popen because frozen Python source copies lacked executable mode, before any request/service launch. Restore executable mode on those owned temporary copies only; source bytes, oracle and timing unchanged. Both attempts remain retained. Preflight is executability evidence, not a serving result.


## Complete TCP concurrency result

The [full raw](../../bench/results/queue_banks_20261010.jsonl) and
[stdout](../../bench/results/queue_banks_20261010.stdout.txt) finish normally
with EXIT0 at source41e5fc4: 2,621 records, 40 cells, 2,560 requests,
83,200 exact measured output tokens, and ten independent executor starts and
normal exits. Every executor records one host service launch, 8,580 device
model graphs, graph_error=0, 264 publications, zero admission/registration
failures, and eight final FREE slots. All five candidate processes report
banks=8,max_active=8. All five pairs are retained; no failed full-run cell or
post-hoc exclusion occurred.

The [supplementary analysis](../../bench/results/queue_banks_analysis_20261010.txt)
includes its exact script and recomputes oracle agreement, every timestamp
metric, the first-send/last-arrival TPS denominator, prompt balance, process
order, counters and the intervals below. Its first checker mistakenly equated
the stored barrier/future CPU window with the TPS window and failed. Reading
the inherited cell() shows these are intentionally different; the corrected
checker verifies containment and independently recomputes TPS from request
timestamps. Both checker attempts remain visible. Measurement, original
metrics, oracle, timeout, source and all samples are unchanged.

Means are arithmetic means of five paired block values; TPS is the mean of
five cell rates, not a pooled rate. A positive gain means banks-minus-serial
TPS, or serial-minus-banks latency. Intervals are nominal two-sided t95, df4,
conditional on this shared-GPU collection; they have no familywise adjustment
and do not establish an isolated causal magnitude.

| gen64 / eight clients | serial mean | private-bank mean | paired gain | nominal t95 gain |
|---|---:|---:|---:|---|
| **Primary: total output tokens/s** | 230.344378 | 595.069687 | +364.725310 | [+361.080986, +368.369634] |
| **Primary: TTFT ms** | 1829.276558 | 23.656817 | +1805.619741 | [+1789.658322, +1821.581160] |
| Completion ms | 2102.076239 | 857.127669 | +1244.948570 | [+1226.924977, +1262.972163] |
| TPOT ms | 4.330154 | 13.229696 | -8.899542 | [-8.977403, -8.821682] |

Both primary effects are positive in every pair, with nominal intervals above
zero. Mean throughput is 2.58339 times the serial comparator; mean completion
latency falls by about59.2%. However, per-request TPOT increases by about3.06
times. Independent request progression removes waiting behind whole serial
requests and improves aggregate throughput/completion at this concurrency,
while each active request emits subsequent tokens more slowly. This is a
measured tradeoff, not an across-the-board latency improvement. max_active=8
still does not demonstrate simultaneous model-kernel overlap.

| paired block | TPS gain tokens/s | TTFT reduction ms | completion reduction ms | TPOT reduction ms |
|---:|---:|---:|---:|---:|
| 0 | +367.147928 | +1808.813123 | +1251.860902 | -8.840511 |
| 1 | +367.368739 | +1822.292134 | +1264.922017 | -8.847145 |
| 2 | +360.840590 | +1811.665708 | +1245.955405 | -8.979529 |
| 3 | +365.839352 | +1792.134424 | +1232.746949 | -8.879166 |
| 4 | +362.429940 | +1793.193316 | +1229.257578 | -8.951361 |

## Diagnostics and implementation costs

Single-client gen64 effects are inconclusive; none of these intervals excludes
zero. The new scheduling mechanism does not establish faster individual model
computation.

| gen64 / one client | serial mean | private-bank mean | nominal t95 gain |
|---|---:|---:|---|
| Output tokens/s | 229.598173 | 229.462857 | [-2.295026, +2.024394] |
| TTFT ms | 5.441970 | 5.438350 | [-0.248967, +0.256208] |
| TPOT ms | 4.337837 | 4.340241 | [-0.042568, +0.037761] |
| Completion ms | 278.725727 | 278.873535 | [-2.782383, +2.486768] |

| gen1 / clients | metric | serial mean | private-bank mean | nominal t95 gain |
|---|---|---:|---:|---|
| 1 | Output tokens/s | 187.022402 | 184.215714 | [-8.629934, +3.016558] |
| 1 | TTFT/completion ms | 5.330338 | 5.410282 | [-0.244871, +0.084983] |
| 8 | Output tokens/s | 198.996008 | 214.925069 | [+13.245113, +18.613007] |
| 8 | TTFT/completion ms | 37.989774 | 36.986667 | [+0.591449, +1.414763] |

One-token/eight-client means improve modestly, while single-client effects
remain inconclusive. Gen1 has no TPOT. Per-request p99 figures in stdout are
pooled descriptive statistics, not independent block-level tail estimates;
for example gen1/eight-client TTFT p99 rises from41.036071 to45.770807ms despite
the mean reduction. No general tail-latency guarantee follows.

Five bootstrap durations average10.312394s serial versus54.058607s private
banks, with ranges[10.250398,10.401339] and[54.015439,54.108346]s. Loading,
capture, native graph instantiation and readiness are included in this startup
measurement and excluded from steady-state cells. Capturing and instantiating
private graphs is a substantial startup cost.

Across the20 active cells per engine, executor-process CPU totals are0.13s
serial and0.08s private banks, at100Hz/0.01s accounting resolution. Each
engine's five one-second idle samples total0.01s CPU. These coarse values
support sleeping-host behavior; they do not establish zero CPU work, a precise
CPU saving, total host/kernel cost or energy efficiency.

## Shared environment, cleanup and interpretation

[Host residency snapshots](../../bench/results/queue_banks_environment_20261010.txt)
record a separate SGLang scheduler with18,402MiB GPU memory before, during and
after this run. The active-run snapshot also attributes python processes with
13,038MiB and652MiB to this bpf-gpu-server container; total GPU memory is32,127MiB.
This is a coarse residency observation, not a measured peak, an isolated
allocation delta, or a before/after serial-versus-bank memory comparison. The
larger graph footprint is a practical concern on this32GiB device. No other
owner was stopped or modified. Per-cell clocks/temperature/power/utilization
are retained in raw; their endpoints do not establish within-window co-tenant
activity, energy use, or causal isolation.

Post-run diagnosis at2026-10-10T19:46:16UTC finds only the same SGLang GPU
process; this task's executor/benchmark GPU processes have exited. The boot ID
is unchanged. The kernel module and bpfusion pins are absent, and every driver
cleanup call returned successfully; final eight FREE slots were observed before
detaching. Cleanup is recorded in the supplementary analysis and environment
raw. The retained original Workspace, PVC and unrelated setup_env.sh are
preserved.

The five paired blocks support retaining concurrent request progression for
this closed-loop workload and this shared RTX5090 collection. The result is a
supporting engineering comparison against the previous serial resident path,
not an optimized external-serving win. Default-transport correctness is
separately qualified above; the performance table uses the documented common
temporary TCP_NODELAY setting. Fixed isolated speedup, physical-NIC behavior,
open-load capacity, total-host cost, energy, text serving and competitive
serving remain unproven. Historical findings have their own environments and
must not be compared as a longitudinal speed curve.


## Independent result review

A fresh read-only result reviewer independently recomputed oracle agreement,
all request metrics, cell TPS, balanced prompts, rotating cell/process order,
all five-pair t95 intervals, preflight/failures, runtime counters and cleanup.
It checked frozen sources against their reported commits, the common temporary
transport delta, private-bank ownership and parent-tail lifetime semantics,
and the shared-GPU interpretation. Its verdict is valid and supporting for
this recorded component/workload comparison, with no publication blocker.
Both primary improvements and the TPOT/startup/memory costs remain explicit;
no separately saved reviewer report is claimed.

The implementation commit has net30 production lines and net13 test lines
(17 test additions /4 deletions). This result milestone adds only necessary
raw evidence and paragraph-level documentation; no production helper, mode
flag, dependency, framework, queue ABI or kernel module change. The next
engineering decision is to target model/Graph execution costs that still
limit single-request TPOT and make concurrent requests emit more slowly,
while retaining the unchanged strict oracle. Repeating the closed KV or
dispatch comparisons would not resolve that uncertainty.

## Next mechanism diagnostic: device-launched model-kernel activity

The full concurrent comparison and retained RoPE/RMSNorm improvements leave a
specific uncertainty: max_active8 and8 model graphs in flight do not show
whether model kernels actually overlap on the GPU. The current parent tail
joins each round of child graphs before advancing requests. Observing their
execution intervals can guide the next throughput/latency implementation:
overlap would direct attention toward computation/memory and batching costs;
complete but non-overlapping activity would support investigating launch and
scheduling serialization. Missing activity remains inconclusive. Neither
outcome completes the service goal or changes a paper RQ; no paper RQ exists.
This is supporting mechanism diagnosis for the user's performance goal,
not a new serving-speed result. Existing host-replay event counts and
coarse GPU utilization cannot answer it, so this adds distinct evidence.
Repeating closed KV/dispatch/fusion comparisons would not resolve it.

Scope: only temporary instrumentation and its necessary record; production
source/native/module/queue/protocol/oracle and other workloads remain unchanged.
All execution occurs in the original bpf-gpu-server Workspace on current main
fc58faa285c7d169733fe626c0c4ddef288d3cb8. The temporary adapter
/tmp/bpfusion-qwen-pointwise-5mxeys8g/device_trace_preflight/qwen_graph
is the exact retained source plus32 instrumentation lines, using the existing
default native library. It discovers Torch's loaded CUPTI/CUDA-runtime paths;
there is no second library installation,driver change or persistent helper.

The installed Torch2.14.1+cu130/CUPTI130001 provides the required process-local
APIs. A capability process returns CUPTI_SUCCESS for device-graph collection
before CUDA initialization,cuInit(0),and HW trace after driver initialization,
without creating a Torch CUDA context or launching model work. The
[CUDA13.0 CUPTI activity API](https://docs.nvidia.com/cupti/13.0.0/api/group__CUPTI__ACTIVITY__API.html)
documents this ordering and Blackwell HES support. Standard concurrent-kernel
activity does not serialize kernels. Actual trace coverage still needs the
real preflight; API success alone is not evidence of device-launched activity.
No newer Torch annotation API or GPU-wide counter profiler is used.

The adapter enables those APIs before model bootstrap,captures the existing
eight banks unchanged,records source graph IDs/node counts,and wraps native
serve with installed torch.profiler.profile CPU/CUDA activities. It exports
the standard Chrome trace as trace.json.gz via export_chrome_trace. The
profile covers native graph instantiation/upload,live requests and normal
shutdown; analysis must exclude setup and the persistent dispatcher when
assessing model-kernel activity. Source graph IDs are checked against exported
activity before any per-bank attribution. Unmapped IDs or dropped/missing
activity are explicit coverage limits.

Reuse oracle_cases,request,connect,cell and command from the completed
comparison driver. The parent builds independent eager-fp16 expected tokens
before starting the child. A one-request/gen1 real TCP/HTTP preflight checks
correct bytes,normal exit,one publication/model launch,zero graph errors,
FREE slots,cleanup and exported model activity. If the installed exporter
cannot expose device model events,record that failure rather than interpreting
an empty trace as serialized compute.

After usable preflight,collect one integrated diagnostic with two separately
started profile processes:clients1 and8,each8 requests generating64 tokens,
four fixed prompts twice each. Both use identical source/native/default
server transport; the existing client TCP_NODELAY setting is common. No
warmups or requests are silently omitted. Each process should report8
publications,512 model graphs,one host launch,zero errors,eight FREE slots
and cleanup; record actual max_active. Existing request/shutdown60s handling
and readiness-or-process-exit remain unchanged. Stop only this experiment's
executor through its native signal/cleanup path after clients finish.

The primary observation is overlap of timestamped model-kernel execution
intervals,using CUPTI's start/end semantics. Report event/category/name counts,
available graph/stream IDs,invalid timestamps and observable coverage first.
Exclude dispatch,API,upload,copy/memset and unrelated setup events from
model-kernel overlap calculations; retain them in raw. Report peak concurrent
model-kernel intervals and union time with at least two model kernels active,
separately for each client condition. These diagnostic summaries have no
paired significance or serving-capacity interpretation. Request timing under
profiling is recorded only to validate the path; it is not a new latency/TPS win.

A positive overlap observation supports that narrower activity claim under
profiling. Absence supports non-overlap only with complete executed-model
coverage; otherwise the result is inconclusive. Intervals do not establish
simultaneous SM instruction issue,isolated GPU behavior,energy or a competitive
serving advantage. Co-tenant residency,boot identity and cleanup remain recorded.
Preserve the adapter,command,oracle/client raw,profiler raw and analysis path
in cited ordinary artifacts. Fresh plan and terminal result reviews apply;
there is no control schema,extra production flag or separate report framework.

The concrete driver is
/tmp/bpfusion-qwen-pointwise-5mxeys8g/device_trace_probe.py.
Run HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/device_trace_probe.py --preflight,
then the same command without --preflight for both full cells. It saves
bench/results/qwen_device_graph_trace_preflight_20261010.txt or
qwen_device_graph_trace_full_20261010.txt with the exact adapter/driver,
independent oracle,client rows,executor output,counters and cleanup.
Compressed profiler traces remain in the corresponding OS-temp directories
until coverage analysis,then the cited retained full traces enter bench/results/.
No duplicate production source or benchmark control framework is retained.

Fresh read-only plan review admits this diagnostic with no scientific or
executability blocker. It checked the exact adapter/driver,oracle independence,
native signal/cleanup,installed API ordering and gzip exporter. Catalogue
cudaGraphGetNodes totals include all node types, so they cannot alone prove
kernel coverage; inspect category/graph/stream attribution before interpreting
non-overlap. The client1 control distinguishes intrinsic graph activity from
cross-request overlap, and incomplete device activity remains inconclusive.

The real preflight terminates EXIT0 with one exact token,one publication,
one model graph,one host launch,error0 and eight FREE slots. Its standard
export contains808 model-kernel records with nonzero timestamps and two
dispatcher records. Source graph5 has809 total nodes,while exported execution
graph ID is1529; do not assume these IDs are the same. Before the full run,
the temporary adapter adds seven CPU-only catalogue lines to record each
source graph's node-type counts. This repairs coverage accounting before
profiling begins; model/native/request/oracle/profiler/timeout paths remain
unchanged. Full instrumentation growth is39 lines,production0/test0.
Original preflight source is preserved in its raw; full raw records the
updated source. No timing comparison uses the catalogue or bootstrap work.
The container nvidia-smi compute-process query returns empty output despite
allocated GPU memory; co-tenant/process observations therefore require the
host namespace and are unknown from that empty container query.

## Device-launched model-kernel activity result

The full diagnostic terminates EXIT0 in both client conditions, without
warmup omissions or request exclusions. Each condition completes eight real
TCP/HTTP requests, two occurrences of each fixed prompt, with512 independently
expected greedy tokens. Final counters show eight publications,512 model
graph executions,one host service-graph launch,error0 and eight FREE slots.
Observed max_active is1 and8, respectively. The exact updated adapter,
driver,oracle,client rows,executor output and cleanup are preserved in
[full raw](../../bench/results/qwen_device_graph_trace_full_20261010.txt)
and its [stdout](../../bench/results/qwen_device_graph_trace_full_20261010.stdout.txt).
The [preflight raw](../../bench/results/qwen_device_graph_trace_preflight_20261010.txt)
and [stdout](../../bench/results/qwen_device_graph_trace_preflight_20261010.stdout.txt)
retain the initial adapter and its one-token path.

Unchanged standard profiler exports are retained as
[one-client trace](../../bench/results/qwen_device_graph_trace_c1_20261010.json.gz),
[eight-client trace](../../bench/results/qwen_device_graph_trace_c8_20261010.json.gz)
and [preflight trace](../../bench/results/qwen_device_graph_trace_preflight_20261010.json.gz).
The [analysis raw](../../bench/results/qwen_device_graph_trace_analysis_20261010.txt)
contains the exact CPU analyzer and terminal output. Imported oracle/client
helpers are retained in the
[previous adapter raw](../../bench/results/qwen_norm_tail_adapter_20261010.txt);
command,cpu_seconds,request,connect and cell are also byte-for-byte identical
to bench/resident_dispatch.py at recorded commitfc58faa. The independent
oracle import is tests/resident_qwen.py at that commit. Recreate the recorded
temporary source layout from those files to repeat collection. CPU-only
reanalysis below reads the archived exports and the EXECUTOR_LOG blocks
already retained in full raw,with no dependency on remaining OS-temp logs:

```python
from pathlib import Path
import re
base=Path('bench/results')
raw=(base/'qwen_device_graph_trace_full_20261010.txt').read_text()
logs=re.findall(r'^EXECUTOR_LOG (.*?)(?=^STATS )',raw,re.M|re.S)
assert len(logs)==2
analysis=(base/'qwen_device_graph_trace_analysis_20261010.txt').read_text()
source=analysis.split('ANALYSIS SOURCE\n',1)[1].split('\nANALYSIS OUTPUT\n',1)[0]
source=source.replace("d=root/('device_trace_c'+str(clients));log=(d/'executor.log').read_text()",
                      "log=logs[[1,8].index(clients)]")
source=source.replace("gzip.open(d/'trace.json.gz','rt')",
                      "gzip.open(base/('qwen_device_graph_trace_c'+str(clients)+'_20261010.json.gz'),'rt')")
exec(compile(source,'archived-trace-analysis','exec'))
```

This substitutes input paths/log extraction only; numerical analysis is
unchanged. No production source or benchmark-control interface is added.

Coverage uses kernel-node counts, not all-node totals. The source catalogue
records808 kernel nodes for each executed prefill shape and760 for each
executed decode shape, plus one memset per model graph. Final bank prompt
lengths and counters imply eight prefill and504 decode executions. Both
exports contain exactly389,504 model-kernel records across512 execution
graph IDs, with the exact per-ID count multiset808x8 and760x504. There are
no invalid model timestamps; the events identify GPU0/context1. This checks
the observable executed-model kernel coverage; it is not an invented mapping
from source graph IDs to execution IDs or request-bank ownership. The513
gpu_memset activities per condition,gpu_memcpy activities,setup and dispatcher
records are retained in raw and excluded from the model-kernel interval
calculation. Tensor-copy CUDA kernels captured inside the model graphs remain
model-kernel records and are counted; the copy exclusion refers to the separate
activity category,not those kernels.

| diagnostic condition | one client | eight clients |
|---|---:|---:|
| Correct requests / exact tokens | 8 /512 | 8 /512 |
| Expected / observed model-kernel records | 389,504 /389,504 | 389,504 /389,504 |
| Executed model graphs | 512 | 512 |
| Peak overlapping model-kernel intervals | 1 | 8 |
| Peak distinct execution graphs in those intervals | 1 | 8 |
| Union of model-kernel intervals ms | 1532.792109 | 775.127768 |
| Union time with at least two distinct execution graphs ms | 0 | 564.881573 |
| Fraction of model-kernel union with cross-graph overlap | 0 | 72.875930% |
| Sum of individual model-kernel durations ms | 1532.792109 | 2773.266717 |

The arithmetic sorts start/end intervals, processes ends before starts at
identical timestamps, tracks distinct execution IDs and checks balanced
completion. The eight-client trace has real cross-execution-graph model-kernel
interval overlap under profiling; the one-client control does not. This
resolves the earlier uncertainty beyond max_active counters. It does not
demonstrate simultaneous SM instruction issue,per-bank attribution,whole-GPU
utilization,isolated behavior or a new uninstrumented serving-speed result.
The overlapping durations must not be added as elapsed wall time or compared
with earlier performance collections as a speedup.

The two largest named cuBLAS GEMV groups account for1064.376770ms of summed
individual model-kernel durations with one client and1677.159697ms with eight,
about69.44% and60.48% of their respective summed model durations. These are
duration shares, not wall-time or whole-GPU percentages. The names alone
do not identify particular projection layers or tensor shapes. Actual
cross-graph overlap directs the next engineering investigation toward
model computation,weight/memory reuse and batching efficiency; it does not
support assuming the current child graphs are completely serialized.

[Host-namespace observations](../../bench/results/qwen_device_graph_trace_environment_20261010.txt)
before and after the diagnostic record the same boot and the same other
owner's SGLang process with18,404MiB residency. That owner remains untouched.
Post-run inspection finds no owned profiler/executor processes,loaded module
or bpfusion pins; final FREE slots and successful cleanup are recorded.
The shared GPU prevents isolation and energy claims. Analysis preparation
initially exposed a syntax error before execution; the raw discloses its
correction before the final CPU-only analysis,without changing traces,
requests,oracle or collection. Production0/test0; the temporary full adapter
adds39 instrumentation lines. This is supporting mechanism diagnosis for
the performance goal,not a separate paper RQ or a competitive-serving result.

A fresh independent read-only result reviewer recomputes both full traces,
oracle agreement,coverage and interval arithmetic. Every execution-graph/
graph-node pair is unique; no duplicate model-node records or invalid
timestamps are found. It also checks final counters/cleanup,the before-run
catalogue deviation,source/helper reconstruction and byte-identical archives.
Verdict: valid and complete; cross-graph activity overlap supported in this
profiled workload; supporting engineering value,no scientific or publication
blocker. Complete serialization is contradicted within this scope. No
separate saved reviewer report or independent serving-speed result is claimed.
The documented CPU archive replay also terminates EXIT0 with the same
counts and interval results.

## Next computation diagnostic: projection and kernel-shape attribution

The completed device trace proves model-kernel overlap but identifies its
dominant GEMV groups only by kernel name. The merged gate/up candidate is
closed: its eight-client primary effects do not support retention. This
leaves a decision-relevant uncertainty: do the large duration groups come
mostly from the vocabulary head, MLP projections, attention projections or
attention matrix operations? The answer changes which computation to optimize.
Existing raw traces cannot identify model operations or shapes from names
alone, so a standard shape-aware profiler adds distinct evidence. This is
supporting engineering diagnosis within the unchanged performance goal;
there is no paper RQ or competitive-serving hypothesis to invent.

Use installed torch.profiler CPU/CUDA activities, record_shapes=True and
record_function labels around the existing nn.Linear forwards. The
[official profiler API](https://docs.pytorch.org/docs/2.14/profiler.html)
documents shape recording and trace export; CPU annotation adds labels without
changing tensor expressions. Installed FunctionEvent.kernels and cpu_parent
provide the operation-to-kernel association. Record actual tensor shapes,
projection names, associated kernel names/counts/durations and unmatched
CUDA records. Attribute only associations observed in this eager forward.
Do not infer captured/device-launched ownership from a host association or
declare bandwidth saturation from names/durations.

All work uses main8a83c63 in the existing bpf-gpu-server Workspace. Production,
native/module/queue/oracle/transport and other GPU owners remain unchanged.
The temporary script reuses current qwen_graph.py load/capture logic, cuts
off before native serve, and captures one private bank. It then adds Linear
labels after capture, prepares the same token pattern and causal/KV prefix
as the prior graph diagnostic, and profiles eager forwards for graph
indices5,64,80,128,190: prefill5/64 and decode positions16/64/126.
For each shape compare its argmax token with replay of the unchanged
captured graph at identical input/KV prefix. This is a computation identity
control, not a replacement for the existing full independent strict oracle.
Require an associated CUDA record for every actual Linear module. Preserve
all raw operation rows and standard Chrome exports; report coverage and
unmatched activity before duration attribution.

Runnable source: /tmp/bpfusion-qwen-pointwise-5mxeys8g/projection_shape_probe.py.
The real computation preflight runs python3 SOURCE --preflight (decode16);
the full command omits that flag and completes all five declared shapes.
The script does no module/BPF attach or native serve. Normal process exit
releases its context and graphs; inspect owned processes and host co-tenant/
boot identity around collection. No GPU-global counter session is used.
Results go to bench/results/qwen_projection_shape_{preflight,full}_20261010.txt
and corresponding standard trace archives, with source/command/analysis.
No arbitrary serving deadline, concurrency, precision or persistent flag is
introduced. Profiler wall timings are not compared with earlier TCP runs.

The primary outcome is observed projection/operation-to-kernel association,
its coverage and per-shape duration grouping. If the large GEMV groups map
to a specific projection family, investigate that family; if spread broadly,
favor cross-request weight reuse/batching over one local projection fusion.
Missing correlations remain inconclusive. Durations in this host-driven,
profiled, shared-GPU diagnostic cannot establish uninstrumented speedup,
device-tail scheduling behavior, isolated cost or physical bandwidth.
Fresh plan and terminal result reviews apply; no extra control schema/helper
or independent dependency contribution is introduced.

Fresh read-only plan review admits the current83-line source without a
scientific or executability blocker. It verifies installed direct correlation,
CPU parent labels,absence of hierarchical duration double-counting,
same input/position/KV prefix for eager/captured replay,all Linear labels,
one-bank/no-native resource scope and non-projection attention preservation.
Name/duration counters check coverage rather than unique event identity;
use retained standard trace correlation/time for unresolved records.
The result must not transfer eager duration shares to the earlier device
trace or infer bandwidth/serving-speed benefits from these observations.

The real decode16 preflight exits0 with169 Linear labels and matching
eager/captured argmax315. Its direct CPU associations match all761 physical
kernel/copy/memset records in the standard trace. The initial observer also
counted CUDA user-annotation ranges as CUDA activity,creating unmatched
projection/compiler labels in its all-CUDA counter. These are annotations,
not missing physical kernels. Before full collection,the temporary script
excludes is_user_annotation from that counter and records the property.
This repairs activity classification only; computation,input,KV,shape matrix,
identity control and profiling remain unchanged. The initial source and
annotation records remain in preflight raw; no preflight rerun is needed.

## Projection/operation attribution result

The preflight and full processes both exit0. Full collection completes all
five shapes,with169 actual Linear modules associated in each and eager/
captured argmax controls12095,374,315,785,315 respectively. The controls
compare only the selected token at these inputs; they do not prove complete
logit equality or replace the independent strict oracle. No shape is omitted.

[Full source/output](../../bench/results/qwen_projection_shape_full_20261010.txt)
and [preflight source/output](../../bench/results/qwen_projection_shape_preflight_20261010.txt)
preserve the computation and installed model configuration. The standard
exports are [prefill5](../../bench/results/qwen_projection_shape_5_20261010.json.gz),
[prefill64](../../bench/results/qwen_projection_shape_64_20261010.json.gz),
[decode16](../../bench/results/qwen_projection_shape_80_20261010.json.gz),
[decode64](../../bench/results/qwen_projection_shape_128_20261010.json.gz),
[decode126](../../bench/results/qwen_projection_shape_190_20261010.json.gz)
and [original preflight](../../bench/results/qwen_projection_shape_preflight_80_20261010.json.gz).
All archives are byte-identical to the original exports.
The [CPU analysis source/output](../../bench/results/qwen_projection_shape_analysis_20261010.txt)
reads retained archives directly,not temporary traces. Extract its SOURCE
block through OUTPUT and run from repository root to reproduce the arithmetic.

All physical kernel/copy/memset records match direct CPU operation associations
by name/duration multiplicity,with no unmatched or extra association records.
The totals are809,881,761,761,761. CUDA user annotations are preserved in the
standard exports and excluded from physical coverage. Direct kernel lists
avoid counting parent-inclusive durations repeatedly; the multiplicity match
is a coverage check rather than a unique correlation identity proof.

| graph / count / position | physical CUDA records | MLP gate+up+down us | vocabulary head us | all copy operations us | summed CUDA duration us |
|---|---:|---:|---:|---:|---:|
| 5 /5 /0 | 809 | 575.874 | 172.865 | 314.209 | 2116.776 |
| 64 /64 /0 | 881 | 645.309 | 188.321 | 324.354 | 2176.608 |
| 80 /1 /16 | 761 | 641.725 | 170.273 | 244.283 | 1846.069 |
| 128 /1 /64 | 761 | 634.848 | 175.009 | 245.375 | 1863.395 |
| 190 /1 /126 | 761 | 635.012 | 168.992 | 246.880 | 1874.116 |

These are sums of individual durations in profiled eager forwards on a shared
GPU,not wall-time shares of the earlier device-launched TCP trace. MLP
projections account for34.76%,34.07%,33.88% in the three decode snapshots;
the vocabulary head is approximately9%,while all copy operations are about13%.
The actual operation rows preserve Linear and attention bmm shapes and associations.
For tensor-list operations such as cat,the standard trace retains nested
Input Dims that profile.events().input_shapes represents as empty lists;
use the archived Chrome data for those details.
Decode GEMV variant6 has97 records:72 MLP projections,one vocabulary head
and24 attention bmm operations. Variant7 has96:24 each of query,key,value
and output projections. The same kernel name spans multiple computation
families,so selecting just the vocabulary head from that name would be wrong.

This points toward broad computation/weight-reuse efficiency and specific
avoidable tensor movement. In particular,the installed eager attention
repeats the two KV heads across seven query-head groups before each matmul.
Rows with [1,2,7,context,64] copy shapes make that movement visible; total
copy time includes other copies,so the entire13% is not attributed to KV
repetition. Grouping query heads into rows against their original KV head is
a different next mechanism from the closed merged MLP projection. It may
reduce repeated KV materialization, but reshaping/matmul algorithms may change
rounding and must pass the unchanged exact oracle before any speed comparison.

[Host snapshots](../../bench/results/qwen_projection_shape_environment_20261010.txt)
record the same boot and other owner's SGLang process732929 with18,404MiB
residency before/after; that owner is untouched. Owned profiler processes,
module and pins are absent after exit. Production0/test0,temporary83-line
instrumentation only. Fresh terminal result review is complete; no serving
speed,isolated bandwidth or new independent paper contribution is claimed.


Fresh independent read-only result review finds this diagnostic valid,
complete and publishable within its scope. It joins every physical GPU record
to the exact CPU operation by standard-trace External id,checks unique GPU
correlations and per-operation kernel multisets,and verifies Linear label
containment in CPU annotation ranges. There is no hierarchical duplicate
association or kernel misattribution. All169 Linear shapes match trace Input
Dims,all48 attention bmm rows remain,and all five coverage/duration totals
recompute. The preflight annotation repair changes observation only.
Verdict: attribution supported,supporting engineering value,computation/
mechanism boundary. Broad projection efficiency and reuse remain appropriate
next targets; the vocabulary head alone does not explain cost. No new
serving result or bandwidth diagnosis follows from this collection.

## Causal qualification for preserving exact tensor batching

The existing ordinary HF batch2/4/8 result fails7 of33 groups; independent
batch1 graph banks pass but perform separate projections,while operation
diagnosis identifies substantial projection cost. A next serving design needs
cross-request computation/weight reuse. The load-bearing uncertainty is
which batched arithmetic boundary breaks the fixed greedy reference.
Without that evidence, another wholesale batching candidate repeats an
unlocalized failure. This qualification is a dependency within that serving
development,not a standalone paper contribution or a speed experiment.

Use the same actual eager-fp16 model,exported weights,ten prompt-edge cases,
gen64 independent HF batch1 oracle and33-group batch2/4/8 plus mixed5 matrix.
The oracle is computed before any patch. The published ordinary-batch result
is the unchanged negative control; load_model AST and installed versions
match,so do not rerun it merely to add another run. No retained pointwise
fusion is installed by load_model; this intentionally targets that published
ordinary computation before qualifying a future production graph candidate.

Three native HF component ablations locate the batch-shape effect:
linear_slice calls the original torch.nn.Linear.forward separately for each
batch lane,then concatenates; attention_slice similarly calls the original
HF eager_attention_forward for each lane with its own query/key/value and
correct shared-or-per-lane mask; both combines the two. All other HF
operations,cache,precision,weight/bias,scale,softmax/cast/dropout and64-token
generation stay unchanged. Exact sources and commands are frozen in
[adapter](../../bench/results/qwen_batch_causal_adapter_20261010.txt).
Original functions are saved before patching; hook engagement counts are
recorded per group. The host invokes inference for this diagnostic only;
no steady worker,mode,production source or infrastructure is introduced.

Small real preflight runs greeting/case2 at batch2 for each of the three
ablations:three groups384 generated tokens. Full comparison runs all99
groups29568 generated tokens,including failures; no early stop or oracle
change. Native output includes all actual sequences,first mismatches,hook
counts,installed versions and terminal status. The existing actual token
comparison remains the metric; EXIT0 proves matrix completion only.
Sources and outputs remain ordinary bench/results artifacts.
Command from repository: HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_batch_causal.py
(add --preflight for the separate preflight artifact).

This is a numerical component comparison,not a performance baseline.
Primary evidence is exact sequence agreement versus the independent
oracle,with per-variant/group/lane first mismatch and total counts.
attention_slice qualification would admit batched projections with
per-lane attention as a concrete candidate for captured/native integration;
linear_slice qualification alone directs work toward preserving per-lane
projection arithmetic while investigating standard batched-GEMV reuse.
Only both passing indicates an interaction or multiple sensitive boundaries;
both failing leaves norm/RoPE/cache/other computation as unresolved.
These outcomes localize an implementation boundary but do not identify
specific CUDA algorithms or guarantee a speed benefit.
Any chosen graph/TCP implementation still needs its own qualification and
complete matched serving experiment; this diagnostic is not that acceptance.

PyTorch's [numerical-accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations)
explicitly distinguishes batched versus sliced numerical results.
It supplies rationale,not a prediction of this model's outputs.
Shared GPU/greedy fp16/prompt scope remain limits; no latency,TPS,energy,
NIC,SOTA or whole-RQ result is claimed. Preserve the other GPU owner,
record before/after host residency/boot,normal process exit and absence
of this task's module/pins. Fresh plan and terminal result review apply.
Run only after the ongoing norm_prepare full collection exits and releases
its resources; preserve its experiment unchanged.

Fresh read-only plan review admits this dependency qualification. It
independently verifies prior33-group control,current load_model/oracle_cases
AST identity,installed versions,actual hook engagement and lane/mask/KV
shape/order. All99 groups and all failure rows are required; no serving
performance claim follows. A both failure also leaves sliced tensor strides
and algorithm selection unresolved,not only other model operations.
No production/test/native source changes or new persistent setting are made.

Real preflight terminates0 with all three groups/384 generated tokens and
actual hook engagement:each Linear-sliced group10816 calls,each attention-
sliced group1536 calls. linear_slice matches greeting/batch2 exactly;
attention_slice and both differ at output62 (actual11,expected323) in both
lanes. This is expected diagnostic contrast,not an execution defect or a
complete qualification result. The full99-group matrix now begins with
the frozen source/oracle/metrics unchanged.

### Complete slicing qualification: no complete batching candidate

The [full raw](../../bench/results/qwen_batch_causal_20261010.txt) exits0
and contains all99 groups/29568 generated tokens. Each variant completes its
33-group matrix. Generated tokens are not all exact; matrix completion
does not qualify any variant. [Analysis](../../bench/results/qwen_batch_causal_analysis_20261010.txt)
preserves its exact CPU script and recomputes every token comparison,first
mismatch,case/batch/group balance and actual hook counts.
All33 groups per variant engage exactly10816 outer Linear-slice wrapper calls
and/or1536 outer attention-slice wrapper calls according to their branch.
Original per-lane function calls multiply those counts by batch size. The three
[preflight](../../bench/results/qwen_batch_causal_preflight_20261010.txt)
rows are byte-value-identical to the corresponding full rows.

| computation | passing groups /33 | failed groups | failed lanes | generated tokens |
|---|---:|---:|---:|---:|
| published ordinary HF control | 26 | 7 | 31 | 9856 |
| original Linear per lane | 25 | 8 | 44 | 9856 |
| original attention per lane | 25 | 8 | 32 | 9856 |
| both per lane | 23 | 10 | 50 | 9856 |

| variant | failed batch2 groups | failed batch4 groups | failed batch8 groups |
|---|---|---|---|
| linear_slice | 4 | 2,4,mixed5 | 0,2,4,mixed5 |
| attention_slice | 0,2,4,mixed5 | 4 | 2,4,mixed5 |
| both | 0,2,mixed5 | 0,2,mixed5 | 0,2,4,mixed5 |

Full failure token sequences and per-lane first mismatches remain in raw.
Important contrasts show why one successful preflight cannot qualify the
route:linear_slice recovers greeting/batch2 and arithmetic/batch4,but newly
fails one-token edge/batch2,greeting/batch4 and France/batch8.
attention_slice recovers arithmetic/batch4 but newly fails France/batch2
and the one-token edge/batch2. Both slicing recovers the one-token edge at
batch2/4 but still fails France/greeting/mixed groups.
For example linear_slice greeting/batch4 first differs at output5
(actual13,expected0),where the ordinary control passed;both slicing
greeting/batch4 differs at output62 (actual11,expected323).

No variant satisfies the complete fixed numerical reference.
Do not install a slicing production path,collect serving speed for it,or
add a fallback/precision/oracle change. These contrasts expose sensitivity
to batch decomposition but do not uniquely identify a CUDA algorithm,
stride,normalization,RoPE,cache or other mathematical cause.
Even sliced operations can keep different tensor layouts/strides and
receive already-divergent activations. The appropriate next implementation
investigation is a fixed-history activation/operation comparison that
locates the first numerical divergence before another batching candidate,
not repeating this completed matrix or assuming both slicing is sufficient.
This remains dependency-only engineering evidence,not an independent
serving/paper result.

[Before/after host observations](../../bench/results/qwen_batch_causal_environment_20261010.txt)
retain the same boot and SGLang owner732929/18404MiB.
Post-run cleanup at2026-10-10T22:36:17UTC records no owned qualification
process,module or BPF pins. This host-driven numerical diagnostic never
loads the TCP module; it does not verify worker-free service or transport.
Production/native/test growth remains0. Full starts at a98a316; the later
5b7a036 only publishes norm records and preserves the same computation.
Fresh independent read-only terminal result review recomputes every actual
token,first mismatch,group/lane count,all preflight/full rows and hook
engagement. It confirms complete valid execution,contradicted sufficiency
of these slicing candidates and inconclusive unique root-cause localization.
Research value is dependency-only,paper impact is the mechanism/workload
boundary. Source/loader/oracle/versions/control identity,lane/mask/KV order,
original function restoration and cleanup are independently verified.
The historical control is a source/numerical control,not randomized
performance evidence. No invalidating defect is found. Close this matrix,
retain all failures,and do not promote any variant or relax the reference.

## Fixed-history first-difference diagnostic for tensor batching

The completed99-group slicing qualification supplies new contrasts but no
complete candidate. Switching Linear/attention boundary changes failures
non-monotonically; another whole-model patch would still lack a causal
discriminator. The next uncertainty is the earliest observed model boundary
where batched and independent batch1 arithmetic differ when token histories
are held identical. This dependency supports actual shared-weight batching
development under the unchanged user performance goal; it is not a new RQ,
standalone serving experiment or speed result.

Use unchanged real eager-fp16 weights/loader and independent oracle computed
before hooks. Reuse the complete ordinary and slicing raws as numerical
controls. Choose three existing contrasting conditions:case2/greeting at
batch2 through output62,case4/one-token prompt at batch4 through output31,
and case0/France at batch8 through output13. Their ordinary and sliced first
greedy mismatch positions are already measured. Include all four original
computations:ordinary,linear_slice,attention_slice,both. No new arithmetic
candidate,precision or production fusion is introduced.

For each case,first execute a batch1 reference one prefill then ordinary
incremental decode with fresh DynamicCache. At every step,feed the previous
independent oracle token. Require all reference predictions to match that
oracle. The candidate jobs similarly begin with fresh independent lanes and
feed the identical oracle history,even after a prediction diverges. This
prevents different generated histories from masquerading as a computational
cause. It is teacher-forced diagnosis,not actual generated serving sequences.

Use standard torch.nn.Module forward pre/post hooks on Embedding,Linear and
Qwen2RMSNorm,plus the RotaryEmbedding outputs. A transparent wrapper records
HF eager attention's query/key/value/mask/output/scores while calling the
selected original/sliced function without changing its math. Capture tensor
shape,stride,dtype and an immediate CPU copy of lane0 (or shared first lane).
Do not retain GPU tensor owners or clone tensors on-device; hooks return None
and never replace module inputs/outputs. Original sliced functions remain
the previously completed implementations. Verify matching tap order and
compare CPU values byte-for-byte. This is the first differing *instrumented
boundary*,not the first CUDA instruction or every internal operation.

Full matrix:three cases109 reference steps plus four variants436 steps,
545 actual model forwards. Preserve every step,all candidate lane predictions,
different-boundary labels/counts,first differing label/shape/stride and
first token-prediction difference. Record whether each condition reproduces
its prior first greedy difference within the observed prefix.
If instrumentation fails that identity control,the corresponding inference
about the original execution is inconclusive;keep its raw,do not silently
substitute a different tensor or expected position.
Preflight uses greeting/batch2 prefill only:one reference and four candidate
forwards. It establishes real hooks,data path and output executability;
it does not qualify the full matrix or establish a root cause.

Ordinary logs retain every comparison summary. Save the actual CPU arrays
for the earliest differing boundary and immediately preceding boundary in
a standard compressed NumPy npz archive,with names linked from raw. These
are synthetic-public-input activations,not model parameters/weights.
Independent CPU review can recompute those stored comparisons;all other
tap equality summaries remain instrument-reported,not an independently
archived full activation trace. This avoids claiming deeper fidelity than
the retained evidence supports. No SHA/checksum gate or control framework.

The result determines which specific component/input-layout must be examined
before another batch implementation. Equal inputs but unequal outputs at an
instrumented module support examining that module's arithmetic/layout;an
earlier unequal input prevents blaming its later output. A matching prefix
with differing token choice later prioritizes the first recorded divergent
step/boundary. No observed difference leaves this diagnosis inconclusive.
None of these outcomes qualifies a Tensor Core/batched-GEMV or TCP candidate
or establishes bandwidth,speed,total CPU,energy,NIC or SOTA superiority.

Exact source/commands are frozen in
[adapter](../../bench/results/qwen_batch_first_difference_adapter_20261010.txt).
Commands from repository root:
HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/trace_batch_first_difference.py --preflight
then the same command without --preflight.
Preflight/full logs and their standard arrays archives use distinct
qwen_batch_first_difference filenames in ordinary bench/results.
Run after the preceding completed qualification has released resources.
Record source/version,host residency/boot and normal cleanup;preserve the
other owner and unchanged production/native/test source. Fresh plan and
terminal result review apply. Shared hardware,hook synchronization,
lane0 observation,limited chosen prefixes and module-level boundaries
limit causal interpretation;this host diagnostic is not worker-free service.

Fresh read-only plan review admits the fixed-history diagnostic. It
independently confirms contrasted prior failures,109+436 forward counts,
independent oracle,teacher-forced fresh KV,actual transparent hooks,
matching tap order and immediate CPU-only snapshots.
The earliest boundary can be harmless rounding rather than the unique cause
of a later token difference. A prior passing condition's reproduction means
agreement only in its observed prefix. NPZ permits independent checks only
of the saved earliest/previous pair;other tap summaries and lane0 scope
remain explicit. No additional gate or baseline is needed.

Real preflight completed normally with EXIT0:one independent reference and
four candidate forwards,all next-token predictions matching the oracle.
For all four variants the first observed difference was prefill step0 at
model.layers.9.input_layernorm.output,with2 differing bytes and maximum
absolute difference0.0009765625;the immediately preceding norm input was
byte-identical. Reference shape1x5x896 and candidate2x5x896 had lane stride
4480,896,1. This establishes the executable diagnostic and a module-level
numerical boundary only,not the full matrix or the cause of later greedy
choices. The preflight preserves16 CPU arrays in its NPZ archive.

Before full execution,the first SSH observation timed out without launching
an experiment. Tailscale ping subsequently answered;real SSH returned the
same lab boot04dc0087-5ccc-4c46-9997-b9a92f0e5167 and Kubernetes reported
Ready. No reboot,service interruption or runtime configuration change was
performed. This transient management-path failure did not restart a run.

### Completed fixed-history result

Full collection reached BATCH_FIRST_DIFFERENCE_COMPLETE and EXIT0:
109 batch1 reference forwards plus436 candidate forwards,12 complete conditions,
1464 recorded candidate lane predictions and584 taps per forward. All reference
checks match the independent oracle. CPU analysis independently reconstructs
all lane prediction differences,verifies12/12 prior first-greedy-prefix identities
and recomputes all44 saved earliest/previous arrays (299246-byte NPZ).
Preflight's16 arrays are also recomputed. No condition or step is excluded.

| Case / batch | Variant | First observed boundary (step) | Different bytes / max abs | First token difference |
|---|---|---|---|---|
| 2 / 2 | ordinary | model.layers.9.input_layernorm.output (0) | 2 / 0.0009765625 | 62 |
| 2 / 2 | linear_slice | model.layers.9.input_layernorm.output (0) | 2 / 0.0009765625 | None |
| 2 / 2 | attention_slice | model.layers.9.input_layernorm.output (0) | 2 / 0.0009765625 | 62 |
| 2 / 2 | both | model.layers.9.input_layernorm.output (0) | 2 / 0.0009765625 | 62 |
| 4 / 4 | ordinary | model.layers.0.self_attn.q_proj.output (0) | 5 / 0.00048828125 | 31 |
| 4 / 4 | linear_slice | model.layers.5.self_attn.core.output (4) | 43 / 1.52587890625e-05 | 31 |
| 4 / 4 | attention_slice | model.layers.0.self_attn.q_proj.output (0) | 5 / 0.00048828125 | 31 |
| 4 / 4 | both | none in observed prefix | none | None |
| 0 / 8 | ordinary | model.layers.0.self_attn.q_proj.output (0) | 770 / 0.015625 | None |
| 0 / 8 | linear_slice | model.layers.0.post_attention_layernorm.output (0) | 1 / 0.000244140625 | 13 |
| 0 / 8 | attention_slice | model.layers.0.self_attn.q_proj.output (0) | 770 / 0.015625 | None |
| 0 / 8 | both | model.layers.0.post_attention_layernorm.output (0) | 1 / 0.000244140625 | 13 |

All candidate lanes share the recorded first prediction-difference index in
these selected conditions. None means agreement only in the observed prefix.
For every saved norm/projection pair,the saved input is byte-identical and
output differs. For case4/linear_slice,the immediately preceding saved pair
is the attention mask,not all attention inputs;Q/K/V equality there follows
from the instrument-reported matching earlier tap summaries,not separately
archived Q/K/V arrays. Case4/both has no observed tensor difference across
its32-step prefix. The full trace therefore identifies multiple observed
component boundaries:normalization,projection and attention.

Early byte differences are not necessarily sufficient to change greedy output:
case2/linear_slice differs at the same norm boundary but does not change token
choice in its63-step prefix;case0/ordinary has770 differing bytes at Q output
but no prediction difference in14 steps. Conversely,case0/linear_slice and
both first differ by one byte at post-attention norm and later change output13.
Do not infer a unique cause,causal necessity/sufficiency or error probability
from these associations. This is finite teacher-forced,lane0 boundary diagnosis;
it does not qualify autoregressive tensor batching,captured/shared-KV math,TCP,
serving speed,bandwidth,energy,total CPU or optimized-serving superiority.
Only the saved earliest/previous comparisons are independently recoverable;
all other activation equality summaries and reference-oracle assertions remain
instrument/source-reported. This is dependency-only evidence guiding the next
implementation,not a standalone research contribution.

Raw [full](../../bench/results/qwen_batch_first_difference_20261010.txt),
[preflight](../../bench/results/qwen_batch_first_difference_preflight_20261010.txt),
[analysis](../../bench/results/qwen_batch_first_difference_analysis_20261010.txt),
[full arrays](../../bench/results/qwen_batch_first_difference_arrays_20261010.npz),
[preflight arrays](../../bench/results/qwen_batch_first_difference_preflight_arrays_20261010.npz)
and [environment](../../bench/results/qwen_batch_first_difference_environment_20261010.txt)
retain commands and finite scope. Actual source is frozen in the adapter.
After collection,the owned process is absent;lab boot is unchanged and the
same SGLang owner732929/18404MiB remains. Production/native/test source is
unchanged;no candidate is promoted by this diagnosis.

Fresh independent terminal result review finds the run valid and complete,
with no invalidating defect. It independently recomputes all candidate lane
predictions,prior prefix identities and44 full plus16 preflight arrays,and
confirms transparent hooks,fresh KV,fixed history and source preservation.
Localization to multiple observed boundaries is supported;unique later-token
causation remains inconclusive. Research value is dependency-only,with a
mechanism/workload boundary and no speed or serving qualification claim.
The next decision is a distinct arithmetic/layout implementation based on
these boundaries,not another unchanged diagnostic or oracle relaxation.

## Batch mean-reduction qualification with component interaction controls

The completed first-difference diagnostic identifies a new numerical boundary
under fixed token history:equal norm inputs can produce unequal outputs.
Other selected variants first differ at a projection or attention boundary.
The installed Qwen RMSNorm forward performs float32 conversion,pow2,last-axis
mean,rsqrt,multiply,fp16 cast and weight multiply. Installed CUDA Reduce.cuh
computes block dimensions from both reduced elements and output count.
This motivates testing original per-lane mean,not declaring a unique cause
or repeating the already completed unchanged33/99-group matrices.

Question under the unchanged user throughput/latency objective:does preserving
batch1 mean reduction while retaining the original other operations recover
complete fixed-reference tensor batching,alone or with the previously tested
Linear/attention component controls? Planned role:dependency for actual
shared-weight device-scheduled batching,not a standalone paper RQ,speed result,
competitive baseline or mathematical root-cause proof. Existing raw outputs
cannot answer the new mean intervention. Direct TCP integration currently lacks
a numerically qualified batched candidate,so this is the strongest runnable
step toward weight reuse;another small pointwise fusion would not resolve it.

Use the unchanged real eager-fp16 exported weights,load_model and independent
HF batch1 oracle_cases(64,True),computed before patches. No precision,oracle,
model,device-owner or service change. Reference ordinary33 groups and completed
three slicing ablations are historical numerical/source controls only,not
contemporaneous hardware performance baselines.

The new norm forward preserves original float32 conversion,pow2,rsqrt,epsilon,
multiply,fp16 cast and weight multiply. It changes only mean's batch call:
take each squared tensor[i:i+1],call the same mean(-1,keepdim=True),and
concatenate its float32 variance. Batch1 calls the saved original norm.
This is a numerical implementation candidate,not an optional runtime setting.
Potential additional launch/concatenation costs are not interpreted as a
performance gain. Slice tensor layout and CUDA algorithm selection may still
differ from independent batch1;successful token qualification is empirical
for the complete tested matrix,not proof of bitwise intermediate identity.

One integrated matrix,four new variants:
mean_slice (original batched Linear and attention);
mean_linear_slice (same mean plus original per-lane Linear);
mean_attention_slice (same mean plus original per-lane eager attention);
mean_both (same mean plus both component controls).
Keep the previously completed Linear/attention slice definitions unchanged.
Each variant resets original-or-sliced functions explicitly,no cross-variant
patch carryover. Every group starts fresh KV,prefill then63 incremental
autoregressive decode steps. Candidate actual selected tokens feed its own
next step;this qualifies generated sequences,not teacher forcing.

For each variant,run batch2,4,8 across all ten existing prompt cases including
prompt-length edges,plus the existing same-length mixed France/greeting group:
33 groups per variant,132 groups,total39424 generated tokens and8448 forwards.
Compare each lane's complete64-token output with the independent fixed oracle.
Preserve actual sequences,pass/failure,first mismatch and engagement counts
for every group. No exclusion,alternate oracle,reduced generation or precision
change. mean_slice engagement should be3136 outer norm calls per group;
the existing Linear/attention controls should record10816/1536 outer calls
when enabled,with original per-lane calls additionally proportional to batch.
These counts are mechanism checks,not performance measurements.

Real preflight uses greeting/B2 for all four new variants:four full64-token
groups,512 generated tokens. It verifies the actual model,oracle,path and
logging. Numerical mismatch is a preserved qualification result,not an
executability fault or permission to change the reference;the full matrix
is necessary to test sufficient recovery and component interactions.

Terminal collection requires BATCH_MEAN_SLICE_MATRIX_COMPLETE and EXIT0,
all132 unique groups and all64-token lane outputs. Independently CPU-recompute
every mismatch/count from raw. Process success alone is not qualification.
For each variant,complete33-group exact agreement supports the tested numerical
candidate;any failed group contradicts sufficiency. Select only a fully qualified
variant for later captured/shared-KV and strict TCP testing,prioritizing variants
that retain batched projections/attention. A variant requiring all components
per-lane is a correctness control rather than evidence of useful weight reuse.
No qualified variant means no promotion;use observed failures to decide the
next distinct implementation or close this path,never rerun unchanged matrices.
Partial recovery,new failure and mixed outcomes remain visible.

Exact runnable source and commands frozen in
bench/results/qwen_batch_mean_slice_adapter_20261010.txt.
From repository root:
HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_batch_mean_slice.py --preflight
then the same command without --preflight.
Use separate ordinary preflight/full raw filenames and analysis output;
record installed versions,source identity,host boot/residency and cleanup.
Run after the preceding diagnostic process ends;preserve the original Workspace,
PVC,other SGLang owner and all unrelated WIP. Production/native/test source
remains unchanged for qualification. Fresh plan and terminal result review apply.
Shared GPU,synthetic short prompts,one installed math stack,finite test coverage,
component layouts and no TCP/captured runtime limit the inference.

Fresh independent read-only plan review admits this new mean-slice qualification.
It confirms original math boundaries,independent oracle,unchanged prior component
slice ASTs,fresh autoregressive KV,132/39424/8448 matrix counts and engagement,
source/analysis executability and the finite numerical qualification scope.
No invalidating defect was found. Preflight/full identity is checked as an
observational stability control;any disagreement must remain visible rather
than selecting one output. No additional baseline or precision setting.

Real preflight completed with BATCH_MEAN_SLICE_MATRIX_COMPLETE and EXIT0,
four generated64-token groups/512 generated tokens. Numerical outcomes are
preserved per variant;collection success is not full qualification. All132
planned groups now run to test sufficient recovery and component interactions.

### Completed mean-slice qualification result

Full collection completed BATCH_MEAN_SLICE_MATRIX_COMPLETE and EXIT0,all132
groups and39424 generated tokens,with no exclusion. CPU analysis reconstructs
every actual lane's first mismatch and pass status,all engagement counts,
full group/case/lane identity and preflight/full row identity. Preflight is
four64-token groups/512 generated tokens:mean_slice and mean_attention_slice
fail greeting/B2 at output62 (11 versus323),while mean_linear_slice and
mean_both pass. These preflight rows match their full counterparts exactly.

| Variant | Groups passing / total | Failed lanes | Generated tokens |
|---|---|---|---|
| mean_slice | 23 / 33 | 48 | 9856 |
| mean_linear_slice | 25 / 33 | 42 | 9856 |
| mean_attention_slice | 23 / 33 | 38 | 9856 |
| mean_both | 33 / 33 | 0 | 9856 |

Only mean_both passes the complete33-group fixed autoregressive reference,
9856 exact generated tokens,including all prompt edges and mixed lanes.
The preceding both-sliced control passed23/33 with50 failed lanes;adding the
per-lane mean intervention removes all ten prior failing groups without a
new failure in this matrix. This supports finite sufficiency of the combined
implementation,not mean alone or a unique internal CUDA reduction cause.
All variants contain the same mean intervention,so the other three failing
variants contradict sufficient recovery while retaining one or both original
batched components. Do not count all39424 tokens as exact.

Full actual outputs,mismatch identities and all changes versus ordinary remain
in [full](../../bench/results/qwen_batch_mean_slice_20261010.txt) and
[CPU analysis](../../bench/results/qwen_batch_mean_slice_analysis_20261010.txt).
The original computations may show partial recovery and new failure rather
than monotonic improvement:mean_slice restores arithmetic/B4 but introduces
France/B2,France/B8,greeting/B4 and mixed/B4 failures relative to ordinary.
Preserve these outcomes;pass totals alone do not identify a mathematical cause.

The qualified mean_both is a correctness control:Linear and attention still
call the original function separately for every lane. It establishes no
useful projection weight reuse,batched Tensor Core/GEMV execution,CUDA Graph
capture,shared-KV or TCP speed improvement. Production/native/test source is
unchanged and no default implementation or service setting is promoted.
The next development question is whether a real batch-matrix projection
can preserve lane-local matrix geometry and exact output with these known
mean/attention boundaries,rather than merely concatenating per-lane Linears.
Such a new computation needs its own fixed-reference qualification.

[Preflight](../../bench/results/qwen_batch_mean_slice_preflight_20261010.txt),
[frozen sources](../../bench/results/qwen_batch_mean_slice_adapter_20261010.txt)
and [environment](../../bench/results/qwen_batch_mean_slice_environment_20261010.txt)
retain reproducibility. Terminal CPU cleanup confirms no owned qualification
process,module or map pins. The same lab boot and SGLang732929/18404MiB owner
remain. No hardware,precision,oracle,workspace or persistent entity change.
This numerical dependency result does not establish worker-free serving,
competitive superiority,error probabilities,total CPU,energy or real NIC results.

Fresh independent terminal result review confirms valid complete collection,
all132 groups and39424 generated outputs,all first mismatches and engagement,
preflight/full identity and unchanged source/oracle. mean_both alone passes
33/33 and9856 exact tokens;finite combined sufficiency is supported,while
mean-only or single-component sufficient recovery is contradicted.
No invalidating defect. Research value remains dependency-only;the next
decision is retain this correctness control and test a distinct actual
batch-matrix projection,with no default or serving-speed promotion.

## Shared-weight batched-matrix projection qualification

The completed mean matrix supplies a fully qualified numerical control:
mean_both passes33 groups/9856 exact tokens,but each original Linear/attention
function is still called per lane. Mean alone or retaining one batched component
does not qualify. This cannot achieve the intended efficient weight reuse.
The next question under the unchanged worker-free throughput/latency objective
is whether a standard batched-matrix projection can replace per-lane Linear
while preserving each request's NxH matrix geometry. Planned role is dependency
for real device-scheduled batching,not a new paper RQ or speed contribution.

Use official torch.bmm for unbiased projections and torch.baddbmm with the
original bias and default alpha/beta for biased projections. Instead of flattening
B*N rows as ordinary Linear does,inputs remain BxNxH;the transposed original
weight is expanded as a zero-batch-stride view to BxHxO. Assert shared original
data pointer and zero view stride;no weight clone is authored. This establishes
logical storage sharing only,not physical kernel reuse,cache traffic,Tensor Core
selection or performance. The implementation may internally copy/change math.
Batch1 or non3D input calls saved original Linear. No out_dtype,backend,precision,
model,epsilon,oracle or persistent setting override. References:
[official bmm](https://docs.pytorch.org/docs/2.14/generated/torch.bmm.html),
[official baddbmm](https://docs.pytorch.org/docs/2.14/generated/torch.baddbmm.html).
Installed function docs are frozen with the candidate;none guarantees bitwise
identity with Linear/addmm,particularly bias/reduction and algorithm selection.

Keep the already examined per-lane original mean implementation unchanged.
Two variants:bmm_original_attention,bmm_sliced_attention. The latter retains
the original per-lane attention control;the former keeps original batched
attention. Reuse mean_linear_slice and fully qualified mean_both raws respectively
as historical numerical/component controls;ordinary and earlier slicing evidence
are already complete. Do not rerun them as pretend randomized speed baselines.
The new matmul implementation is genuinely untested by those prior raws;
running it directly in TCP would lack qualification,while another unchanged
mean/Linear/attention matrix adds nothing. This candidate can either unlock
a lower-launch batched projection or reject a standard API route before service
integration;both decisions change the next engineering action.

Independent HF batch1 oracle_cases(64,True) precedes all patches;unchanged real
eager-fp16 exported weights/load_model. Fresh per-group prefill and KV,then63
incremental decode steps using the candidate's own selected tokens. No teacher
forcing,no sharing KV lanes,no changed expected output after failure.
Full matrix:two variants x batch2,4,8 x all ten existing cases plus mixed5,
66 groups/19712 generated tokens/4224 forwards. Preserve full actual64-token
lane sequences,first mismatches,passing flags and mechanism counters.
Every group expects10816 batched Linear/shared-weight-view outer calls,
6208 bmm,4608 baddbmm and3136 per-lane mean outer calls;attention_slice1536
only when enabled. matrix_rows must equal168*(prompt_length+63)+64,demonstrating
logical per-request N geometry,not device performance. The168 layer Linears
use all prefill rows,but logits_to_keep=1 makes the vocabulary head use one
row for prefill as well as each decode;the greeting's total is11488.

Real preflight uses both variants on greeting/B2,64 autoregressive tokens each:
two groups/256 generated tokens. It tests real model,inputs,oracle,mechanism
and logging. If both variants already fail that unchanged qualification,
they are both ineligible:close at that explicit numerical rejection branch,
preserve all raw,mark the66-group expansion unexecuted and claim no completed
full matrix. Do not spend an unchanged full matrix on variants already rejected
without a decision-changing diagnostic question. If at least one variant passes,
run the planned full66 groups to terminal status,including failing rows.
Do not change any arithmetic,oracle,timeout or evaluation during either run.

Full collection must reach BATCH_BMM_MATRIX_COMPLETE and EXIT0,with all66
unique groups and actual64-token outputs. CPU analysis independently reconstructs
all mismatches,pass/group/case identity,counters and preflight/full stability.
Process exit0 is not numerical qualification. A variant's complete33-group
exact agreement permits later graph/shared-KV/strict TCP testing;any failed group
rejects that variant. Finite passed prompts do not establish arbitrary correctness,
rounding equality,performance or actual device-side weight reuse. Preserve all
partial recovery,new failure and mixed output rather than choosing a passing
prefix. No qualified variant means no default promotion or relaxed reference.

Source/commands frozen in bench/results/qwen_batch_bmm_adapter_20261010.txt.
From repository root:
HF_HOME=/workspaces/.cache/huggingface python3
/tmp/bpfusion-qwen-pointwise-5mxeys8g/qualify_batch_bmm.py --preflight
and,only for the admitted expansion branch,the same without --preflight.
Separate preflight/full raw files,ordinary CPU analysis,installed versions,
host boot/residency and terminal cleanup. Original Workspace/PVC/other owner
and tools/setup_env.sh are preserved. Source remains temporary;production/native
and tests unchanged. Fresh plan and terminal result review apply.
Shared GPU,one stack,finite synthetic prompts,API/internal arithmetic changes
and no graph capture/TCP/performance measurement limit interpretation.

Fresh independent plan review admits this batched-matrix candidate and its
explicit rejection/expansion branch. Before any model execution,it identified
and corrected the logical-row expectation:168*(prompt_length+63)+64,not
169*(prompt_length+63),because logits_to_keep=1 restricts the vocabulary head.
The actual counter and candidate math were unchanged;the frozen analyzer uses
the corrected formula. Review confirms real Linear engagement,independent
oracle/fresh autoregressive KV,API/bias semantics,control reuse and scope.
No remaining invalidating defect. Zero stride/pointer equality establishes
logical sharing only;qualification remains an empirical question.

### Shared-weight BMM terminal numerical rejection

Both real greeting/B2 preflight groups terminate normally with full64-token
autoregressive outputs,256 generated tokens total. Both variants fail both
lanes at output index62:actual11 versus fixed reference323;the preceding62
tokens agree and index63 also differs. The corresponding historical
mean_linear_slice and mean_both controls pass this row. Each candidate group
records10816 actual batched Linear/shared-weight-view calls,6208 bmm,4608
baddbmm,3136 mean and11488 logical matrix rows;the sliced-attention variant
also records1536 attention calls.

This follows the predeclared rejection branch:both unchanged candidates are
ineligible and the full66-group/19712-token expansion was not executed.
Do not claim a full matrix or256 exact tokens. The default implementation
is preserved;no TCP/capture/performance collection or physical weight-reuse
claim follows. This rejects only these concrete API replacements,not every
possible batch-matrix implementation or the worker-free performance objective.

[Preflight raw](../../bench/results/qwen_batch_bmm_preflight_20261010.txt),
[frozen sources](../../bench/results/qwen_batch_bmm_adapter_20261010.txt),
[CPU analysis](../../bench/results/qwen_batch_bmm_analysis_20261010.txt) and
[before/after environment](../../bench/results/qwen_batch_bmm_environment_20261010.txt)
preserve all actual sequences,mismatches,engagement and numerical controls.
CPU analysis independently reconstructs each mismatch and verifies the
explicit full-matrix absence. Terminal cleanup confirms no owned qualifier,
module or pins. Before/after boot remains04dc0087-5ccc-4c46-9997-b9a92f0e5167,
and the other SGLang owner732929/18404MiB remains. No production/native/test,
precision,oracle,hardware or persistent configuration change.

Fresh independent result review judges valid terminal numerical rejection,
contradicted qualification for these two variants,dependency-only research
value and no invalidating defect. It independently checks frozen/actual source,
oracle-before-patch,fresh KV/self-selected decode,all64-token sequences,
historical controls,counters,full66 absence and cleanup. Logical sharing is
not physical reuse;the rejection does not identify a unique matmul/bias/backend
cause. Move to a distinct computation mechanism that can improve real TCP
throughput/latency rather than repeat rejected unchanged qualification.

## Upstream MPK SM120 compatibility and terminal numerical rejection

The upstream-compute investigation follows the requested separation between
model computation and the Linux/eBPF request path. Mirage revision
f9eb70c254acefc9f3667b2a973d0dcf25471fce was built in an OS-temporary environment
inside the original Workspace. The model is Qwen3-0.6B revision
c1899de289a04d12100db370d81485cdf75e47ca, BF16, on the shared RTX5090.
The production executor, model math and existing oracle were preserved.

Three defects were identified before model qualification could finish:
the Ampere attention task reads a ten-field parameter ABI while the selected
Python path supplied six fields; the selected CUTLASS LM-head three-stage
pipeline needs110592 shared bytes against98304 allocated; and shared A/C
storage reuse lacks synchronization. The temporary patch supplies the existing
attention ABI, uses the upstream two-stage pipeline on SM120 and adds two CTA
barriers. It also selects existing Ampere RMSNorm/linear tasks and the SM120
shared-memory budget. These are compatibility/correctness changes, with no
new model-arithmetic implementation or serving-default promotion.

The original LM-head memcheck reports6081 errors. Two stages remove the
out-of-bounds access but retain numerical failure and a two-CTA racecheck
reports9592 errors/3256 warnings. With synchronization, the two-CTA racecheck
reports zero hazards/errors/warnings. A separate full600-CTA memcheck reports
zero errors and both constant and dyadic-random cases match all1228800 output
elements. The preserved two-CTA random-case label misleadingly says all600
tiles; its actual launch/check covers two tiles, and full coverage belongs
only to the separate sync_full probe.

The complete patched online-pinned model runs three waves of four requests,
then shuts down without reported kernel errors. All12 prefixes fail the
unchanged32-token HF greedy oracle at output index0. A further diagnostic
submits, completes and releases one request at a time, retaining the same
compiled kernel, capacities and weights: all12 still fail at index0.
The one-token prompt has identical complete generated output across modes;
the other three prompts differ at the first generated token. Prompt bytes
are preserved in both modes. Each mode generates6072 tokens, which must not
be described as6072 correct tokens. This shows that simultaneous submission
is unnecessary for the observed failure, without identifying a unique
compute, loader, metadata or runtime cause.

[Frozen source, patch, commands, complete outputs, sanitizer receipts and
ordinary CPU analysis](../../bench/results/mpk_sm120_qualification_20261011.txt)
preserve the positive and negative controls. Both harnesses exit0 and stop
their kernel thread, but neither qualifies the model. The10.523978-second
parallel and26.801169-second serial durations include harness/model work;
they are not matched client TTFT, TPOT or throughput results. Explicit cleanup
also triggers an ignored upstream double-finalize assertion at destruction,
retained in the log. No network/TCP integration or performance collection
follows from these failed model results.

The host boot and other SGLang process remain present and the04:25 UTC
journal read has no new NVRM Xid since04:00. Process residency is not service
acceptance. No reset/reboot or persistent policy was changed. MPK remains an
unqualified upstream candidate; further model-specific math tuning is not
admitted as evidence for the eBPF path. A distinct public SM120 backend is the
next integration candidate.

## Upstream Plow SM120 build and mixed numerical qualification

Plow revision [627f09c6990e0d60fa0596b0a8b20bdba1dd5cf5](https://github.com/infervisor/plow/tree/627f09c6990e0d60fa0596b0a8b20bdba1dd5cf5)
was built in OS-temporary storage inside the original Workspace, using its
Rust compiler/runtime, native CMake targets and CUDA13.0.88. The real upstream
Lean4.15.0 verifier was built and used during packet emission; verification
was not disabled. Its certificates cover ordering and scoped rewrite checks,
with explicit gaps for floating-point implementation, complete memory effects
and empirical performance. The emitted packet also warns about impure prefill
segments. These warnings are preserved; prefill was not executed or qualified.

The existing qwen3_sm120_chat harness consumes prompts one token at a time
through its decode program. A temporary compile/link patch guards a cubin-only
debug symbol and makes the referenced device metadata definitions visible to
the host pass with C linkage. It changes30 definitions of19 metadata symbols;
model arithmetic remains upstream. The patch and the compile-only linkage
probe preserve both the initial failures and the successful build.

The first GPU run fails with CUDA719 under memcheck. The native Qwen harness
defaults to GF4, while the unchanged Qwen3-0.6B checkpoint has16 query heads
and8 KV heads. The upstream dispatch explicitly traps when GQA modulo GF is
nonzero. Selecting the existing GF2 arm resolves this compatibility failure.
The corrected one-token memcheck and all four32-token memchecks terminate
with zero reported errors and matching device/host argmax. This is finite
memory-check evidence, not race freedom or complete model correctness.

All four prompts and the32-token HF eager-BF16 greedy oracle are reused
unchanged from the earlier MPK qualification. Arithmetic and Chinese prompts
match all32 generated tokens. The one-token prompt differs at output index0
(21806 versus14582); the France prompt first differs at index12
(5429 versus8585). Thus two of four complete sequences pass;128 tokens were
generated, not128 reference-correct tokens. Ordinary CPU analysis reconstructs
the actual sequences and first mismatches from each raw log.

The existing PLOW_DUMP_LOGITS interface supplies a fixed-history diagnostic
at both failed positions. A fresh HF generate call reproduces the original
oracle prefixes and exposes the corresponding raw logits. Across all151936
logits, relative L2 errors are0.0148267 and0.0178287; KL(reference||Plow) is
0.0009691 and0.0019013. The HF top-two margins are0 and0.0625 respectively.
Plow breaks the first reference tie and introduces a tie at the second
position. These observations explain argmax sensitivity to the observed
numeric changes; they neither locate a unique implementation cause nor
establish general quality equivalence. No new acceptance threshold is adopted.
An earlier standalone-forward diagnostic uses different invocation geometry
and must not replace the generate oracle; both records remain visible.

[Frozen patch, commands, build/emit failures, successful builds, complete
generated sequences, sanitizer receipts and diagnostics](../../bench/results/plow_sm120_qualification_20261011.txt)
provide reproducible evidence. No model-weight, packet or executable binary
is committed. The host boot remains04dc0087-5ccc-4c46-9997-b9a92f0e5167;
SGLang PID2762187 remains resident at18346MiB, and the04:58 UTC journal read
has no NVRM Xid since04:30. Residency is not other-service acceptance.
All owned qualification processes are terminal; no reset/reboot, persistent
configuration, production executor or unrelated WIP was changed.

Plow is a buildable SM120 candidate with mixed numerical results, not a
qualified replacement, resident network integration or serving performance
result. Further model-specific arithmetic tuning would not establish the
eBPF path benefit. Retain the qualified HF graph executor while evaluating
a distinct upstream computation interface or a clearly declared independent
quality contract before integrating an alternative backend.

## Upstream model-as-a-kernel SM120 qualification

Hub v1 revision [5858c6ddbe04c6b749e81daeb47bcd1857c1e618](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel/tree/5858c6ddbe04c6b749e81daeb47bcd1857c1e618)
was built in temporary storage in the original Workspace, with Torch2.14.1+cu130
and CUDA13.0.88 on the shared RTX5090. The importer/binding match the pinned
source bytes; the CUDA file differs only in CRLF/LF line endings. Computation
was unchanged. A temporary registration-header shim enables the upstream
torch.ops.mak_ext fallback with a non-Python extension build. Flags are -O3
and -DCUDA_KERNEL; Torch selects native SM120. No fast-math flag is added.

The initial attempt stops before compilation because Ninja is absent.
Ninja1.13.2 is installed only under the temporary build directory and added
to that invocation's PATH. The retry builds and runs the actual model;
NATIVE_BUILD_PASS and compiler commands are retained in the raw receipt.

The model is HuggingFaceTB/SmolLM2-135M in BF16. The upstream importer consumes
the same loaded weights as the HF eager reference. Four text prompts plus
six repeated-pattern prompt lengths1/15/16/17/32/64 each generate64 tokens.
The oracle prefills the complete prompt, then performs cached greedy decode
without early EOS termination. Each candidate runs single-launch generation,
per-step launches and repeated single-launch generation with reused buffers.

All10 cases agree across the two launch modes and repeat. Seven of ten
complete64-token sequences match HF exactly. The other three first differ
at zero-based output positions10 (eight-token arithmetic prompt),2
(two-token Linux prompt), and0 (one-token pattern). Each of the four output
arrays has64 tokens in every case. Thus determinism and mode agreement pass,
but the unchanged strict reference qualification fails. This does not locate
a unique numerical cause or establish general quality equivalence.

[Complete script, shim, build failures/success, inputs, reference outputs,
candidate outputs and exit receipts](../../bench/results/mak_sm120_qualification_20261011.txt)
are retained. The terminal qualifier exits2 at2026-10-11T06:02:50UTC.
A separate CPU reconstruction checks all10 rows and reproduces7 exact cases,
three first mismatches and all mode/repeat agreements. No sanitizer, network
integration, online admission or matched serving performance was qualified.
No model-specific arithmetic tuning follows from this result. The qualified
HF graph path remains available while broader upstream interfaces are assessed.

## HF graph reuse across Qwen and Llama architectures

The existing graph bootstrap now accepts an HF checkpoint or model directory,
in addition to the original Qwen weight binary. It captures the upstream HF
forward into the existing eight graph/KV banks. The mask adapter passes the
causal tensor accepted by both installed Qwen2 and Llama implementations.
No new model operator, quantization, kernel arithmetic or per-model tuning
was added. Existing Qwen-specific compile wrappers remain unchanged; they
do not modify Llama computation.

The final matrix ran in the original Workspace on the shared RTX 5090 with
Torch 2.14.1+cu130, Transformers 5.19.0 and CUDA 13.0.88. Checkpoints were:

- Qwen/Qwen2.5-0.5B-Instruct, cache revision
  7ae557604adf67be50417f59c2c2f167def9a775: Qwen2, 24 layers, hidden 896,
  14 attention heads, 2 KV heads and vocabulary 151936.
- HuggingFaceTB/SmolLM2-135M, cache revision
  93efa2f097d58c2a74874c7e644dbc9b0cee75a2: Llama, 30 layers, hidden 576,
  9 attention heads, 3 KV heads and vocabulary 49152.

Each used its own independent HF eager-FP16 full-prompt plus cached greedy
reference. Four text prompts and prompt lengths 1/15/16/17/32/64 generated
64 tokens, without early EOS termination. The existing TCP/HTTP regression
also checks eight concurrent clients, rejection followed by valid requests,
mixed generation lengths, repeated slot reuse, pipelining and half-close.

| Final case | Valid requests | Exact generated tokens | Rejections | Empty connections | Host launches after bootstrap | Device graph launches |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM2 resident |335|20432|259|5000|1|20432|
| SmolLM2 host-control |335|20432|259|5000|594|20432|
| Qwen default resident |335|20432|259|5000|1|20432|
| Qwen default host-control |335|20432|259|5000|594|20432|

All four exited 0, with zero graph errors, eight banks and max_active 8.
Each trace includes all executor threads and has no accept/receive/send
syscalls after readiness. All queue slots returned FREE with zero admission
drops; test hooks, map pins and the module were removed. Total final coverage
is 1340 valid requests, 81728 exact generated tokens, 1036 rejections and 20000
empty connections. A separate CPU reconstruction checks all 156 printed
64-token reference/actual pairs, counters, trace boundaries, exits and cleanup;
the harness checks the other response arrays before printing its aggregate.

The qualification found a real shutdown issue in the resident native path:
SIGTERM delivered to a CUDA helper sets the stop flag but does not interrupt
the main thread's long sleep. The host lifecycle wait now checks that flag
between one-second sleeps. This is not a request worker or compute change.
The final test deliberately sends SIGTERM to a non-main executor thread;
the trace confirms all four target threads differ from the readiness writer.
Whole-executor cleanup took 5.542/4.723/4.066/3.591 seconds respectively,
including CUDA destruction; these are lifecycle observations, not latency
benchmarks or a claim of one-second process exit.

Reproduce after building the existing graph targets:

```sh
make qwen-graph qwen-graph-control
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph --model HuggingFaceTB/SmolLM2-135M
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph_host_launch --model HuggingFaceTB/SmolLM2-135M
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph
python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph_host_launch
```

The [raw receipt](../../bench/results/hf_graph_multimodel_20261011.txt)
preserves final source, commands, traces, build output and independent
reconstruction. Earlier attempts include a logging-attribute failure, the
Llama mask-interface failure and a manually assisted old-build shutdown.
The first failed attempt's stale trace is explicitly excluded. Only ATTEMPT4,
2026-10-11T06:26:03--06:34:15UTC, supports the final four-case qualification.

The 06:36:54UTC host observation retains boot
04dc0087-5ccc-4c46-9997-b9a92f0e5167 and SGLang PID 2762187 at 18348 MiB;
no NVRM Xid appears in the journal read since 06:00UTC. Residency is not
other-service acceptance. All owned qualification processes are terminal,
and unrelated setup_env.sh WIP remains untouched.

This qualifies two checkpoint/architecture combinations in the existing
fixed-context, greedy graph interface, not arbitrary HF models, MoE/hybrid
architectures or shared-weight continuous batching. It is FP16 HF graph
reuse, distinct from the earlier BF16 model-as-a-kernel strict-reference
failure. It supplies a second architecture without a model arithmetic
campaign; no alternative megakernel, serving speedup, real-NIC result or
complete userspace-path comparison is established by this matrix.

## AlpinDale RTX5090 Qwen qualification: complete numerical rejection

The pinned MIT backend5030e154d39ecd054df03eb4dd9c8aa8185414d1 builds
with its own JIT/default CUDA flags on the existing RTX5090. Its cached
Qwen3-0.6B checkpoint is c1899de289a04d12100db370d81485cdf75e47ca.
This is BF16 Qwen3 computation, distinct from the current FP16 Qwen2.5 path.

Source inspection before execution finds importer RoPE theta10000 versus
checkpoint theta1000000. The reviewed plan declares both unchanged-theta
and checkpoint-theta conditions. Both retain the upstream CUDA/binding
source, flags, reductions and sampling. The only shared loader compatibility
edit replaces device_map="cuda" with explicit .to("cuda"), avoiding an absent
accelerate dependency. Required ninja1.13.2 is confined to owned OS-temp;
the existing Python environment is preserved.

All planned32 complete64-token sequences finish normally,2048 actual tokens:
four full tokenized prompts, step and bulk APIs, two repetitions and two
theta conditions. Independent HF BF16 eager and default-attention references
are frozen before loading the native backend. Each condition has0/16 exact
sequences against either reference. Step/bulk outputs agree and repeated
reset calls agree; this is stable numerical disagreement, not a missing reply
or partial collection. First eager mismatch indices are[3,0,4,2] for the
four prompts in both conditions. For France, the first actual token changes
from32671 to1112 after theta correction, versus reference12095.
The two HF attention references first disagree later on Hello/France
(indices17/12), so their own differences do not explain the earlier native
mismatches. No EOS truncation, selected suffix or oracle replacement is used.

The native build and mode/repeat checks are dependency evidence. Close this
strict replacement candidate without a CUDA arithmetic campaign or a serving
performance claim. A theta importer correction alone is insufficient; no
cause for the remaining disagreement is established. There is no new
network adapter, production source change, concurrent-safety qualification
or accepted latency/throughput gain.

- [Reviewed plan](../../bench/results/alpindale_plan_20261012.md)
  and [plan review](../../bench/results/alpindale_plan_review_20261012.md).
- [Actual qualifier](../../bench/results/alpindale_qualify_20261012.py)
  and [complete raw with original/adapter sources](../../bench/results/alpindale_qualification_20261012.jsonl).
- [Full stdout](../../bench/results/alpindale_stdout_20261012.txt),
  [native build commands](../../bench/results/alpindale_build_20261012.txt),
  [fresh result review](../../bench/results/alpindale_result_review_20261012.md)
  and [restoration audit](../../bench/results/alpindale_restoration_20261012.json).

Actual invocation:
`HF_HOME=/workspaces/.cache/huggingface python3 /tmp/bpfusion-alpindale-8dv7jm6x/qualify.py`.
For reproduction, reconstruct the pinned source tree from the raw metadata
under a fresh OS-temp directory, install the recorded ninja dependency there
and copy the archived qualifier into that directory. Select a new result
destination in the copy before running; preserve the original archived raw.
The qualifier records its actual source and source-native flags again.
Source/build/dependency apparatus is removed after archival and review.
The original Workspace/cache and unrelated tools/setup_env.sh remain.
