# Chained device-graph dispatch

## Implementation and current status

The existing dispatcher uses one model fire-and-forget launch plus one parent
tail launch per step. The opt-in candidate attaches the unchanged dispatcher
as a dependent node after upstream body/head and tail-launches the chosen next
combined graph. CUDA explicitly permits tail self-launch. Model/head/sampling,
FP16, KV budget, packed prefill, FIFO-prefix release and both transports stay
matched. The ordinary default remains separate: the complete A/B below establishes no reliable primary improvement.

The [plan](../../bench/results/tail_chain_plan_20261012.md) and
[fresh plan review](../../bench/results/tail_chain_plan_review_20261012.md)
admit one supporting causal comparison. Official semantics are documented in
[CUDA device graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#tail-launch).
This is reusable GPU orchestration, not arithmetic fusion or an eBPF-only gain.

## Completed setup and strict qualification

Eight binaries build using the pinned temporary official runtime. Four traced
Qwen preflight executors pass160 requests/4192 independently replayed tokens.
Native successful launch counters show2/model-step in separate mode and1/step
in chained mode; trace step counts independently agree.

Both models/four conditions finish strict checks:4976 requests/177792 valid
tokens,2072 intentional rejects and40000 empty connections. All batch sizes1..8,
fragmentation,edges,wrap/reuse,mixed output lengths,pipelines/half-close,
rejection/subsequent valid response and helper-thread stop pass. Each actual
client sequence matches independently advanced host replay of its actual trace.
After-ready all-thread syscall checks engage native network operations and find
no steady serving accept/recv/send calls in either kernel condition. All native
exits/analysis are0; output release/cache ownership and teardown pass.

Before any untraced observation, the frozen complete-output set at
2026-10-13T00:28:04.805780+00:00 combines earlier finding0034 references and this
run's independently replayed strict/traced-preflight sequences. No untraced
output contributes. Exact whole-sequence membership is a finite-set
qualification, not replay of the current unobserved batch/cache trajectory,
universal numerical correctness, task quality or batch invariance.

The complete untraced matrix finishes once: Qwen driver terminal at
2026-10-13T00:39:48Z, Smol at00:48:46Z, both analyses0 and wrapper0.
All40 executors/160 cells pass10240 measured requests/332800 tokens plus320
checked gen1 warmups excluded from timing. Every reply meets the frozen whole-
sequence contract, every executor exits normally and every terminal page/native
counter recycles its admitted work. No full failure, exclusion or favorable
retry occurs. Actual total10560 requests/333120 tokens includes warmups.
Per-cell untraced graph/batch counts remain null.

## Complete matched client result

Each condition has320 measured requests across five process blocks. Primary
effects are separate-minus-chained gen1/c1 TTFT and chained-minus-separate
gen64/c8 throughput, separately for each model and transport. Five paired
process differences, nominal two-sided t95 df4; pooled requests do not create
additional independent repetitions. All samples and outliers remain.

| Model / transport | Primary | Separate | Chained | Gain favoring chained [nominal paired t95] |
|---|---|---:|---:|---|
| qwen / kernel | gen1/c1 TTFT ms | 1.475003 | 1.437113 | +0.037890 [-0.011647,0.087427] |
| qwen / kernel | gen64/c8 token/s | 4747.698381 | 4647.218259 | -100.480122 [-268.954764,67.994520] |
| qwen / userspace | gen1/c1 TTFT ms | 1.432012 | 1.400747 | +0.031265 [-0.038345,0.100875] |
| qwen / userspace | gen64/c8 token/s | 4689.774039 | 4748.051888 | +58.277850 [-78.796812,195.352511] |
| smol / kernel | gen1/c1 TTFT ms | 1.153345 | 1.171664 | -0.018319 [-0.117197,0.080560] |
| smol / kernel | gen64/c8 token/s | 5495.636932 | 5410.104202 | -85.532731 [-533.342583,362.277121] |
| smol / userspace | gen1/c1 TTFT ms | 1.195710 | 1.192011 | +0.003699 [-0.089045,0.096442] |
| smol / userspace | gen64/c8 token/s | 5580.360946 | 5578.130519 | -2.230427 [-23.766157,19.305303] |

All eight primary intervals contain zero: no reliable candidate gain or
equivalence. Every recorded secondary completion/TPOT/TTFT/TPS and chained
kernel/native transport interval also contains zero. Qwen kernel throughput
mean falls about100token/s and Smol kernel about86token/s; retain those negative
estimates and their broad intervals instead of selecting the positive mean
TTFT entries. Lower device-launch count is not proof of lower end-to-end cost.
The changed graph environment/dependency and actual batch trajectories are
part of the candidate; the effect cannot be attributed exclusively to one call.
Keep ordinary separate dispatch and close this unchanged candidate matrix.

Metrics are measured from actual client send/first/last token timestamps.
Throughput uses the whole cell's first-send-to-last-arrival window. TCP/client
coalescing remains included; capture/bootstrap/connections precede timing.
[Complete Qwen reconstruction](../../bench/results/tail_chain_20261012/qwen_performance_analysis.stdout.txt)
and [Smol reconstruction](../../bench/results/tail_chain_20261012/smol_performance_analysis.stdout.txt)
retain all means,p50/p99,individual paired effects and secondary intervals.
The [Qwen raw](../../bench/results/tail_chain_20261012/qwen_performance.jsonl)
and [Smol raw](../../bench/results/tail_chain_20261012/smol_performance.jsonl)
are the authoritative client evidence. Per-policy output-agreement diagnostics
retain differences between actual transport schedules; membership is not a
claim of identical concurrent outputs.

## Source identity and completion receipts

The qualification source is published as33337adffd0975e7542c359a966cb4a759695742.
Qwen full metadata was emitted while HEAD still reportedfdcd111, with the
candidate diff already present and snapshotted. The qualification commit occurs
at00:30:03UTC before its first measured request at00:30:11UTC; Smol metadata
reports33337ad. Recorded driver/native/backend source is unchanged throughout
the comparison and equals the qualified code. Do not infer a mixed source from
these different HEAD observations or silently replace their actual metadata.

The bounded host journal00:16:00→00:52:08UTC preserves the OOM records and
later activity. NotReady was a documented live API observation; its original
API JSON was not saved and the kernel journal does not itself prove that status. The listed NVRM/allocation/OOM victim lines occur
before full timing; this does not prove continuous host/GPU isolation or absence
of interference. Endpoint receipts show the same boot/kernel/base packages,
original Workspace UID Running/Ready with0 restarts and original Coder PIDs.
No own GPU app,module or pins remain after run. After the
[fresh combined result review](../../bench/results/tail_chain_result_review_20261012.md),
[final restoration](../../bench/results/tail_chain_20261012/final_restoration.json)
removes the owned temporary runtime and32 build artifacts, with the same boot/
base runtime, no own GPU process/module/pins and original Coder/WIP preserved.

- [Actual sources](../../bench/results/tail_chain_20261012/sources.json),
  [commands](../../bench/results/tail_chain_20261012/pipeline_commands.json),
  [fixed references](../../bench/results/tail_chain_20261012/frozen_references.json).
- [Qwen strict](../../bench/results/tail_chain_20261012/qwen_correctness.jsonl),
  [Smol strict](../../bench/results/tail_chain_20261012/smol_correctness.jsonl).
- [Resolved packages](../../bench/results/tail_chain_20261012/resolved_packages.json),
  [bounded journal](../../bench/results/tail_chain_20261012/host_journal.json),
  [owner](../../bench/results/tail_chain_20261012/owner_receipt.json) and
  [Workspace](../../bench/results/tail_chain_20261012/workspace_receipt.txt).

No total service CPU,energy,physical-network,external SOTA,arbitrary-backend or
general eBPF advantage follows. This is supporting implementation/workload
boundary evidence; it does not turn qualification counts into paper novelty.


## Environment and reproducibility

During the first separate-Qwen bootstrap/preflight window lab becomes NotReady
and Tailscale/campus SSH time out. Existing Cloudflare SSH remains usable; the
same boot returns Ready automatically. Global OOM kills other Lean/Coder
processes and iSCSI/EXT4 errors occur on other devices. Original bpf-gpu-server Pod,
Coder processes and pipeline stay live; no replacement/restart/host mutation
occurs. Preserve this unequal setup exposure; do not claim host isolation.

Completed source/commands/build/strict/reference/fault evidence is in
[the experiment archive](../../bench/results/tail_chain_20261012/).
The exact runtime and pipeline commands are recorded there. Reproduce using
the documented vLLM runtime and existing make targets, then
python3 bench/vllm_serving.py --dispatch-comparison with --correctness for
independent traced replay, or --untraced --reference-path FROZEN for the
pre-observation finite-set contract. Analyze the raw with
python3 bench/vllm_serving_analysis.py RAW.
