# Native timed network control and TX task CPU

## Question and scope

Does removing a native TCP request/reply worker reduce CPU charged to the
executor process plus kernel TX task against a CPU-efficient timed epoll
worker, and what client latency/throughput accompanies that difference?

This supporting component comparison retains identical FP16 HF computation,
eight graph/KV banks and resident GPU dispatch. The three conditions are
eBPF/kernel TCP, native busy epoll and native epoll_pwait2 with a60us timeout.
Socket readiness can wake the latter earlier; its effective wait is not
equivalent to kernel usleep_range(60,120), nor claimed optimal.

[Plan](../../bench/results/timed_network_plan_20261011.md) and
[fresh plan review](../../bench/results/timed_network_plan_review_20261011.md)
precede execution. There is no formal paper RQ or external SOTA baseline.
The new condition and TX accounting are absent from finding0026; old raw
outputs cannot recover them.

Run status: complete and valid for the declared supporting component comparison.
The bounded CPU-saving prediction is contradicted against the timed native
control; client latency/throughput remain inconclusive. See the full result
and independent review below.

## Implementation and qualification

Build the new optional control with:
```bash
make qwen-graph qwen-graph-user-network qwen-graph-user-network-wait
make -C module KDIR=/usr/src/linux-headers-7.3.0-070300rc3-generic CC=gcc-15
```

Only the wait library enables BF_USER_NETWORK_WAIT. Existing model math,
GPU dispatcher, parser, partial sends, ordering, rejection and recycling
remain identical. All three dispatcher SASS instruction arrays contain
768 rows and compare equal. Python source compilation and native/module
builds pass. Kernel gcc/pahole version warnings and missing-vmlinux BTF skip
are recorded as build context, not correctness evidence.

The new read-only /sys/module/bfusion_tx/parameters/tx_cpu_ns calls
task_cputime_adjusted for the TX task. An explicit task reference and mutex
preserve getter/shutdown lifetime. The benchmark uses POSIX process CPU
clocks for all executor threads and retains before/after values plus monotonic
read brackets. The sum excludes RX/softirq/BPF and TCP work outside the TX
context, client CPU and GPU computation. It is not total service CPU.

[Strict regression stdout](../../bench/results/timed_network_qualification_20261011.stdout.txt)
records four exit0 commands: SmolLM2-135M and Qwen2.5-0.5B, each under kernel
and wait executors. Totals:1340 valid requests,81728 exact greedy tokens and
1036 rejections. Each condition preserves the unchanged eager-HF oracle,
prompt edges, eight concurrent clients, repeated wrap/reuse, mixed lengths,
HTTP pipelining/half-close, invalid-token recovery, one resident launch,
zero graph errors and cleanup. Tracing distinguishes no steady userspace
network worker in the kernel mode from native userspace network operations
in the controls. These are correctness/lifecycle qualifications, not latency
or throughput conclusions.

[Preflight stdout](../../bench/results/timed_network_preflight_20261011.stdout.txt),
[Smol raw](../../bench/results/timed_network_smol_preflight_20261011.jsonl) and
[Qwen raw](../../bench/results/timed_network_qwen_preflight_20261011.jsonl)
retain six normal executors and24 exact measured one-token replies, with
eight warmup requests/executor separate. Each process reports launch1,
264 model-graph steps, graph_error0 and eight banks. Both native wait markers
engage; kernel TX accounting grows and raw CPU component deltas reconstruct.
Native analysis exits0 for both files. Do not interpret these single-block
four-request cells as performance results.

Qualification/preflight ran on the six reviewed source changes atop bb059df;
a7d097e only added the backend source audit, during preflight. Source was committed before full collection; both metadata records identify
frozen revision9f091f95d6bc359871a10c74077f1d9af35d5caa.

## Environment boundary

[Read-only environment receipt](../../bench/results/timed_network_environment_20261011.txt)
records the unchanged host boot and an external co-tenant transition.
SGLang PID2762187/18368MiB exists at10:27:34UTC. The separate
gpu-smt-candidate-5090-20261007 Job terminates at10:34:41UTC with exit1;
its log reports ENOSPC while appending generation-events.jsonl, after157/186
tasks. Subsequent host reads find no SGLang process. bpf-gpu-server's own32GiB
Longhorn volume has16GiB available; host overlay has346GiB available.
These reads do not identify the cause of the other project's storage growth.

bpf-gpu-server did not stop/restart that Job/SGLang or change its PVC/lifecycle.
The event occurs during strict qualification, before all preflight measured
cells and full collection. Old shared-SGLang performance results are not a
matched comparison for this host condition. No induced isolation or
constant co-tenant/activity claim is made; full-run event exposure must be
reported from current observations.

## Full comparison and interpretation

Source frozen at9f091f95d6bc359871a10c74077f1d9af35d5caa; both metadata
records name that same revision. The two native commands run sequentially
from2026-10-11T10:53:04UTC through11:36:20UTC. Smol exits0 at11:16:19UTC;
Qwen and the full wrapper exit0 at11:36:20UTC. There is no failed, excluded
or selectively retried full-run row.

[Full command stdout](../../bench/results/timed_network_20261011.stdout.txt),
[Smol raw](../../bench/results/timed_network_smol_20261011.jsonl),
[Qwen raw](../../bench/results/timed_network_qwen_20261011.jsonl) and
[native analysis](../../bench/results/timed_network_analysis_20261011.stdout.txt)
retain120 cells,7680 exact measured requests,249600 measured tokens and30
normal executors. All executors report one resident launch,8580 model-graph
steps,graph_error0,eight banks and max_active8. Each receives eight warmups
plus256 measured requests. All150 idle/cell CPU component deltas and sequential
read brackets reconstruct from raw counters. The native analyses both exit0.

The five-pair estimates below use nominal two-sided t95(df4), with no
family-wide or equivalence interpretation. CPU is control-minus-resident
microseconds per request at gen64/client8; negative values favor the control.

| Model | Control | CPU us/request difference [t95] |
|---|---|---|
| SmolLM2-135M | busy native | +67223.325 [66775.581,67671.069] |
| SmolLM2-135M | timed native | -383.283 [-593.022,-173.544] |
| Qwen2.5-0.5B | busy native | +91205.752 [89877.246,92534.257] |
| Qwen2.5-0.5B | timed native | -337.304 [-659.632,-14.976] |

At that primary workload, mean bounded CPU per request is2011.566us resident,
69234.891us busy and1628.283us timed for Smol;2319.134us resident,
93524.886us busy and1981.831us timed for Qwen. The efficient control uses
less measured worker CPU for both models. Kernel TX dominates the resident
counter; this does not identify which internal activity causes its cost.

Standard client metrics accompany the CPU comparison. TTFT is
control-minus-resident at gen1/client1 in milliseconds; throughput is
resident-minus-control at gen64/client8 in output tokens/s.

| Model | Control | TTFT ms difference [t95] | Throughput tokens/s difference [t95] |
|---|---|---|---|
| SmolLM2-135M | busy native | -0.090135 [-0.268859,0.088589] | -6.368173 [-15.373800,2.637453] |
| SmolLM2-135M | timed native | +0.090901 [-0.141011,0.322813] | -2.488979 [-9.443425,4.465468] |
| Qwen2.5-0.5B | busy native | -0.017408 [-0.037196,0.002381] | +3.320190 [-6.397947,13.038327] |
| Qwen2.5-0.5B | timed native | -0.026525 [-0.059269,0.006218] | +2.632198 [-3.308572,8.572968] |

All these client intervals cross zero. This establishes neither superiority
nor equivalence. Mean gen64/client8 throughput is918.517/924.885/921.006
tokens/s for Smol resident/busy/timed and688.077/684.757/685.445 for Qwen.
The busy-loop CPU difference is a tradeoff control, not sufficient evidence
against efficient conventional serving.

[Fresh independent result review](../../bench/results/timed_network_result_review_20261011.md)
reconstructs the complete matrix, oracle, all CPU/timing values and mechanism
engagement. It independently confirms the negative efficient-control comparison
and crossing-zero client intervals. No invalidating defect or rerun is required.

### Event exposure and lifecycle

The full collection starts with a new shared SGLang process observed at
10:51:29UTC,PID4026103,18398MiB, on the same GPU UUID and host boot.
At10:56:58 and11:14:58UTC it remains observed alongside bpf-gpu-server.
Per-cell GPU clocks/temperature/power/utilization are retained; these do
not identify or guarantee constant co-tenant activity.

Retrospective retained kernel journal for the full interval identifies all
ten resident module lifecycles. Each unload reports55176 bytes,264 completed,
zero abandoned/retries,19 accepted and19 closed sockets. Host fault reads
through11:36:30UTC find no matching OOM/Xid/Oops/BUG/Call Trace/I/O/EXT4
fault lines in the requested retained intervals. Absence of those fault lines
does not prove freedom from interference or equal exposure.

The after-run process/GPU inspection occurs later, at21:51UTC. It finds no
benchmark/executor process, bfusion module or pinned BPF object, the same host
boot and an idle GPU. SGLang is then absent. This delayed observation does
not establish when SGLang left, continuous residency, or contemporaneous
after-run GPU conditions; its lifecycle was not changed by bpf-gpu-server.
Do not transfer earlier co-tenant conditions into a claim of isolated data.

### Decision and remaining scope

Run status: valid completed supporting component comparison.
Tested bounded CPU-saving hypothesis: contradicted at gen64/client8 against
this efficient native control for both checkpoints. Client latency/throughput:
inconclusive. Research value: supporting. Paper impact: mechanism/workload
boundary, not a direct challenge to the complete research goal.

Close this matrix and retain all negative results. Do not claim that removing
the userspace worker automatically saves CPU or improves speed, or use the
busy-loop comparison as the headline. The observed difference is not a transport-only causal estimate: shared-host
conditions and CPU accounting boundaries remain competing explanations. Qwen
resident gen64/client8 global CPU work changes from34.59/36.69s in its first
two blocks to17.67/17.02/17.38s later; the cause is not established. The
immediate uncertainty is where kernel accept/drain/completion work is spent relative to the efficient native
control. Source-native profiling could discriminate that cause before a
targeted transport change; the present result alone cannot select such a fix.

Compute reuse remains the intended route. Do not tune model arithmetic to
erase this result or install every surveyed kernel. Physical-NIC behavior,
total attributed CPU, full text/tokenization, online batching and a strong
official serving comparison remain open under the full objective.

## Follow-up CPU localization (dependency evidence)

A separate unchanged-source diagnostic on2026-10-11 samples only the TX task
during the canonical Qwen gen64 strict test's eight-client256-valid/
256-rejection closed-loop stage. The
[qualification log](../../bench/results/tx_profile_qualification_20261011.stdout.txt)
ends normally with335 valid requests,20432 exact tokens,259 rejections,
594 publications,graph_error0,eight banks and cleanup. The pre-measurement
5000 empty connections are outside the sampled stage.

[Native perf report and command receipts](../../bench/results/tx_cpu_profile_report_20261011.stdout.txt),
[raw perf data](../../bench/results/tx_cpu_profile_20261011.perf.data),
[decoded samples](../../bench/results/tx_cpu_profile_samples_20261011.txt) and
[load-time module symbols](../../bench/results/tx_cpu_profile_module_symbols_20261011.txt)
retain1116 software cpu-clock samples,zero lost samples, targeted PID1725252.
Default sample frequency is used with kernel call graphs. Deliberate SIGINT
ends recording after the target stage; perf returns-2 after finalizing data.
Native reports/script subsequently exit0 on that same recording. Builtin
kernel symbols resolve; some module addresses remain unknown after unload.

Exclusive stack counts are327 recv,233 accept,195 send,184 sleep/schedule,
177 other. These are localization diagnostics, not a new paper metric.
Source Linux7.3-rc3 kernel_accept creates a socket before checking the accept
queue, then releases it on failure; the current module calls it every sweep.
kernel_recvmsg also repeats per connected peer even when no input is ready.
The trace supports inspecting these paths, not a claim that all their cost
is wasted or a predicted speedup.

This stage mixes rejected/valid requests, runs under the strict driver's
strace, and is not the primary performance matrix's cell window.
cpu-clock call-stack sampling can include inline softirq/interrupt context
and has overhead; it is not identical to adjusted scheduler TX CPU.
No fraction of total service CPU or causal before/after improvement follows.
This diagnostic has dependency value only. A readiness-check candidate can
use Linux's existing TCP/sockmap readiness semantics to avoid unnecessary
accept/recv calls; its correctness and matched performance remain unproven.
