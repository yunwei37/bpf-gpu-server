# Evaluation and evidence

## Current conclusions

bpf-gpu-server is a working research prototype on its qualified Linux/GPU stack.
Its reusable result is Linux/eBPF connection and request/reply ownership around
caller-supplied GPU execution. General superiority over efficient native TCP,
exclusive total-CPU savings, production SLOs and arbitrary-backend compatibility
are not established. [README](../README.md) describes current scope;
[architecture](architecture.md) describes ownership and adapters;
[getting started](getting-started.md) is the executable runbook.

The LLM speed competition is closed as the main research line by the user's
2026-10-16 decision. The completed
[five-block16/32/official comparison](findings/0080-resident-queue-relief.md)
tests the shared resident capacity repair first. Qwen2.5-0.5B FP16 on RTX5090,
192remote requests/s for60s,64output tokens and five process blocks yield:

| Condition | Mean delivered tokens/s | Mean completionP99 | Mean owned-group CPU/request |
|---|---:|---:|---:|
| Resident native16 | 11639.6 | 3337.27ms | .75066ms |
| Resident native32 | 12265.1 | 217.36ms | .52330ms |
| Official vLLM | 12249.4 | 405.87ms | 8.32700ms |

All172800measured requests complete without observed failures. The32-bank
constructor removes sustained queue growth before first output. It has adverse
mean token time and greater device-wide energy than official. The offered
ceiling is12288tokens/s, so this does not rank maximum capacity. Both resident
transports share capacity/batching/KV changes; these are not eBPF-only gains.
The finding retains paired intervals, finite fidelity flags, full tails,
resource context, source snapshots, raw data and owner-only recomputation.

The [completed bounded small-graph experiment](findings/0081-small-graph-network-cpu.md)
adds60screening cells/2520000requests and30new-seed confirmation cells/
600000requests,all main requests completing without observed failure.
No selected point satisfies the fixed CPU/P99 advantage criterion.
Single-CPU GEMM P99 is adverse (K/N2.589,t95[1.196,5.607]);other selected
CPU/P99 intervals do not establish an advantage or equivalence.
All20path sweeps first saturate the configured setup at16000/s after4000/s.
Shared-host/observer costs,finite numerical patterns and coarse work/load
brackets remain explicit. The original client-timing pair and a later
premeasurement control timeout are preserved separately. Close this bounded
performance search and retain the OSS prototype with its qualified scope.

## Comparators and interpretation

| Comparison | Question it can answer | Boundary |
|---|---|---|
| Kernel-network vs efficient native epoll, same GPU executor | Does transport placement improve this CPU/P99 tradeoff? | Queue backing and Linux/CPU/GPU work placement vary together |
| Resident adapter vs official vLLM | How does whole serving behavior compare at the declared load? | API, scheduler, admission and KV/capacity differ |
| Resident16 vs resident32 | Does common active capacity relieve queuing? | A capacity/capture/KV bundle, not eBPF attribution |
| Legacy module/SQPOLL/GPU-I/O/coalesced/AMK controls | Historical component mechanisms | Separate sources, conditions and acceptance; not current defaults |

Native also retains GPU-controlled execution. It is not artificially forced
through Python per-request inference. The graph-server-uring-loop target is a
GPU-I/O implementation, not an io_uring userspace competitor.

## Measurement contract

Keep every scheduled request, failure, timeout, partial response and client
scheduling delay. Scheduled-arrival response latency includes queue wait for a
busy persistent connection. Begin-to-response is reported separately and is
not pure GPU service time or an exact network-only interval.

Report P50/P90/P95/P99/P99.9/maximum and full empirical distributions,
throughput over offered time plus drain, numerical checks and failure counts.
Successful-only tails condition on completion. Failure-aware ranks put failed
attempts at infinity; an unbounded rank is unknown, not a timeout-valued success.
Do not infer equivalence from crossing-zero intervals.

Resource primaries are owned-process-group CPU/request and scheduled-response
P99. Report CPU/attempt and CPU/success, host user/system/IRQ/softirq, nested
TID/cgroup context, actual CPU masks and adjacent idle measurements separately.
Host totals include foreign tasks; a process-only reduction is not attributed
net CPU saved. Missing namespace/kernel-worker or interval GPU data are unknown.
Client timing guards distinguish client ceilings from server saturation.

The graph calibration uses CUDA events around host graph replay. It includes
host submission gaps and is neither pure device-launch latency nor request
latency. All compared arithmetic stays in unchanged upstream cuBLAS/cuFFT
graphs; no empty/sleep kernel or handwritten math replaces it.

The fixed confirmation requires zero failures, valid resource/timing/mask
evidence and throughput lower95ratio>=.98, plus either CPU upper95ratio<.90
with P99upper<=1.05, or P99upper<.90 with CPUupper<=1.05. Five paired log-ratio
t95 intervals are nominal, selected-cell and small-sample; they carry no
multiple-testing guarantee. Shared-host uncertainty limits any positive claim.

## Repairs and unresolved evidence

Real repairs include finite pending admission, fair native FIFO, connection
reclamation, active waiting and shared TCP_NODELAY. Their numerical/lifecycle
acceptance and adverse measurements live in the findings below. A repaired
precondition is not automatically a transport-speed result. Default8,
ordinary builds and partial-stop qualification are distinct from wider
performance configurations.

The historical Qwen192/s6.36ms receive-gap difference remains unreproduced and
unallocated. Aggregate stage times overlap and cannot be added to invent an
exclusive latency decomposition. Subsequent capacity/NODELAY repairs do not
claim to explain that entire historical gap.

AMK residency/reuse is implemented and finite-qualified in its recorded scope,
but the completed selected comparison is negative. Closed GPU placement,
publication, adaptive-wait and identity-cache candidates remain closed; a
negative new result does not authorize endless variants to rescue a thesis.

## Reproduction and provenance

Each finding points to its plan, exact source/environment, commands, raw
requests/resources, analysis, adverse or failed prefixes and cleanup.
Archived absolute paths identify the original acquisition; portable start,
client and cleanup commands live in the current runbooks.

Current experiments and audits are performed by this sole thread. Earlier
independent-agent reviews are historical artifacts and remain intact.
Owner-only independent-code recomputation is not an independent-person review.
Source freezes, paired order and seeds distinguish real confirmation from
reanalysis. No result is relabeled as freshly verified merely by reorganizing
documentation.

## Historical evidence index

This index replaces duplicated chronological narratives in this page.
Findings and their linked raw archives remain the single homes of exact
historical facts, including original failures, withdrawn acceptance and
unselected candidates. None of those records is deleted by this consolidation.

- [Finding 0001 — ingress → userspace wakeup on this kernel](findings/0001-ingress-wake.md)
- [Finding 0002 — packet → GPU → packet, measured end to end](findings/0002-end-to-end-mlp.md)
- [Finding 0003 — direct-page execution, lost wakeups, and the 20 ms bug](findings/0003-direct-page-and-wakeup.md)
- [Finding 0004 — launch-once resident GPU kernel (no per-request host control call)](findings/0004-resident-kernel.md)
- [Finding 0005 — real Qwen LLM on the pinned page (host-Python executor)](findings/0005-resident-llm.md)
- [Finding 0006 — offline from loopback: the path over a real netdevice](findings/0006-veth-netdevice.md)
- [Finding 0007 — concurrency and the SLO picture of the current executor](findings/0007-concurrency-slo.md)
- [Finding 0008 — CPU instructions per generated token](findings/0008-instructions-per-token.md)
- [Finding 0009 — TCP path (stream request in, stream tokens out)](findings/0009-tcp-path.md)
- [Finding 0010 — matched model-size comparison (0.5B / 1.5B / Qwen3-1.7B)](findings/0010-model-size.md)
- [Finding 0011 — external HTTP baseline (vLLM) vs the bpfusion paths](findings/0011-http-baseline.md)
- [0012: kernel TCP TX correctness control](findings/0012-kernel-tcp-tx.md)
- [0013: real Qwen in one resident CUDA kernel](findings/0013-resident-qwen.md)
- [0014: sockmap TCP stream ingress to resident Qwen](findings/0014-sockmap-stream.md)
- [0015: kernel HTTP token transport to resident Qwen](findings/0015-http-token-transport.md)
- [0016: cooperative resident Qwen and the longer-decode accuracy boundary](findings/0016-cooperative-resident-qwen.md)
- [0017: reject invalid token IDs without ending resident inference](findings/0017-invalid-token-rejection.md)
- [0018: matched resident versus host-dispatch Qwen control](findings/0018-resident-dispatch.md)
- [0019: device-scheduled Qwen graphs and 64-token kernel TCP validation](findings/0019-device-scheduled-qwen.md)
- [0020: matched whole-request dispatch with device-scheduled Qwen graphs](findings/0020-graph-dispatch.md)
- [0021: real-Qwen compute diagnostics and last-position logits](findings/0021-qwen-compute-optimization.md)
- [0022: shared KV append in GPU-scheduled Qwen graphs](findings/0022-shared-kv.md)
- [0023: batch execution qualification](findings/0023-batch-execution-qualification.md)
- [0024: concurrent whole-request graph dispatch control](findings/0024-concurrent-graph-dispatch.md)
- [0025: native userspace network control with reused HF graphs](findings/0025-native-userspace-network.md)
- [0026: HTTP header/first-token coalescing with matched GPU computation](findings/0026-coalesced-network.md)
- [Native timed network control and TX task CPU](findings/0027-timed-network-control.md)
- [Socket readiness before kernel accept and receive](findings/0028-socket-readiness.md)
- [Cross-node TCP/HTTP with reused model graphs](findings/0029-cross-node-network.md)
- [Short inputs still traverse the captured model graph](findings/0030-short-input-latency.md)
- [Official optimized serving comparison](findings/0031-optimized-serving.md)
- [Upstream optimized graphs in the actual kernel-queue server](findings/0032-vllm-kernel-serving.md)
- [Pack mixed-length prefills with unchanged upstream graphs](findings/0033-prefill-packing.md)
- [Release the completed FIFO prefix before new GPU work](findings/0034-completion-release.md)
- [Ordinary untraced optimized serving](findings/0035-default-untraced.md)
- [Chained device-graph dispatch](findings/0036-tail-chain.md)
- [Stock io_uring TCP from an NVIDIA GPU](findings/0037-stock-io-uring.md)
- [eBPF admission and io_uring with real HF model graphs](findings/0038-io-uring-sockmap-models.md)
- [Stock io_uring streaming with real model graphs](findings/0039-io-uring-streaming.md)
- [Multi-connection stock io_uring serving](findings/0040-io-uring-concurrent.md)
- [Stock io_uring with the optimized vLLM adapter](findings/0041-vllm-io-uring.md)
- [Optimized stock TCP: matched delivery and worker CPU](findings/0042-vllm-io-uring-performance.md)
- [Incremental stock TCP output copying](findings/0043-uring-incremental-copy.md)
- [Sleepable eBPF io_uring coordination](findings/0044-uring-bpf-loop.md)
- [CQ progress captured before submission](findings/0045-uring-cq-progress.md)
- [Optimized module-free cross-node serving](findings/0046-optimized-cross-node.md)
- [Current optimized short-input latency localization](findings/0047-optimized-short-input-latency.md)
- [Reused AMK computation in the resident TCP dispatcher](findings/0048-reused-amk-dispatch.md)
- [Additional GPU residency reproduces the AMK c8 loss](findings/0049-amk-residency-counterfactual.md)
- [Coalesced network progression recovers AMK throughput with tradeoffs](findings/0050-coalesced-gpu-network.md)
- [Batched ring publication gives partial gains, not vLLM latency recovery](findings/0051-batched-ring-publication.md)
- [Idle model admission after SQ consumption improves vLLM short-output latency locally](findings/0052-model-admission-after-sq-consumption.md)
- [Completed-response drain before fresh model admission](findings/0053-final-response-drain-before-admission.md)
- [GPU-local I/O bookkeeping: completed throughput discriminator](findings/0054-gpu-local-io-bookkeeping.md)
- [Current GPU I/O arrangements with identical reused model computation](findings/0055-current-gpu-io-arrangements.md)
- [Independent GPU queue publication: no demonstrated primary gain](findings/0056-independent-gpu-queue-publication.md)
- [Long-output GPU service-time diagnostic: incomplete decomposition](findings/0057-long-output-service-timeline.md)
- [Batched token publication: incomplete performance comparison](findings/0058-batched-token-publication.md)
- [Upstream batch-invariant serving: exact finite fidelity, no throughput gain](findings/0059-batch-invariant-serving.md)
- [Adaptive kernel-loop wait: CPU saving with a delivery tradeoff](findings/0060-adaptive-kernel-wait.md)
- [Offered-load admission and tail latency](findings/0061-offered-load-admission.md)
- [Pending admission: complete delivery and mixed service tails](findings/0062-pending-admission-tails.md)
- [No reliable high-load tail advantage against fair native admission](findings/0063-fair-native-admission.md)
- [Sixteen model banks improve delivery; same-width eBPF superiority is unproven](findings/0064-batch-capacity.md)
- [Mixed prefill/decode improves generation time; eBPF superiority is unproven](findings/0065-mixed-prefill-decode.md)
- [Qwen tail-path diagnosis and unsuccessful close-scan repair](findings/0066-tail-path-diagnosis.md)
- [A non-LLM captured-graph transport boundary](findings/0067-opaque-graph-backend.md)
- [Mixed scheduling is available through ordinary source targets](findings/0068-mixed-source-integration.md)
- [Current mixed service versus official vLLM](findings/0069-current-official-vllm.md)
- [Remove ACK-dependent small-send stalls](findings/0070-tcp-nodelay.md)
- [0071: sustained remote load exposes kernel connection failure](findings/0071-sustained-remote-resources.md)
- [0072: repair EOF reclamation under sustained remote load](findings/0072-connection-reclamation.md)
- [0073: creator kernel CPU localizes coordination work](findings/0073-kernel-cpu-localization.md)
- [0074: active admission waiting reduces process-group CPU](findings/0074-active-admission-wait.md)
- [0075: immutable peer identity does not establish a speed benefit](findings/0075-peer-identity-cache.md)
- [0076: current controller timing bounds the remaining speed gap](findings/0076-current-controller-stage-time.md)
- [0077: Kernel-owned network placement, small local gain with higher CPU](findings/0077-kernel-network-placement.md)
- [0078: Official vLLM wins sustained delivery; transport speed advantage unproven](findings/0078-sustained-official-vllm.md)
- [0079: Ordinary kernel-network targets reuse LLM and opaque graph backends](findings/0079-kernel-network-source-integration.md)
- [0080: relieve the shared resident queue with a qualified32-bank profile](findings/0080-resident-queue-relief.md)
- [Bounded small-graph network / CPU comparison](findings/0081-small-graph-network-cpu.md)
