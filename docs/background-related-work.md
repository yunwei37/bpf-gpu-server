# Background and related work

Project decision update,2026-10-17UTC:the
[bounded same-executor library-work confirmation](findings/0081-small-graph-network-cpu.md)
establishes no predeclared useful CPU/P99 advantage. The reusable
Linux/eBPF/graph integration remains an OSS prototype;it is not a newly
established performance contribution. This is an internal-evidence update,
not a fresh literature search. External observations and proposed-next-step
wording below retain their dated historical scope;they are not an active
GPU-kernel/LLM search backlog.


External-source map last updated: 2026-10-15 UTC (scoped reusable-backend refresh below; earlier source checks retain their dated scope).
Source/command: web searches and primary papers, official documentation, upstream source and artifacts; current repository findings.
Completeness: covers GPU networking/proxy removal, GPU-owned Linux I/O, reusable model execution, and recent contradictory performance evidence. This is a scoped map, not an exhaustive bibliography or a new thesis/RQ. No paper contract, idea-story or user-instruction document exists here.

## Search log

| Date | Query/source | Purpose | Result |
|---|---|---|---|
| 2026-10-14 | GPUnet GPU networking CPU proxy OSDI2014; GLoop GPU I/O runtime | Prior GPU server interfaces | GPUnet accepted paper verified; GLoop remains an adjacent lead, not needed to establish the closest overlap |
| 2026-10-14 | Direct I/O from the GPU io_uring; LLVM GPU io_uring | Same Linux-ring/worker-removal mechanism | The early LLVM proposal's cancellation is superseded by its August2025 implemented prototype and public artifact |
| 2026-10-14 | io_uring BPF kernel loop GPU2026; kernel struct_ops source | Upstream enabling mechanism | Linux bpf-ops.c/loop.c expose programmable ring submission/waiting; these primitives are prior work |
| 2026-10-14 | LLM megakernel persistent GPU serving; AMK; Mirage | Reusable computation assets | AMK and MPK official artifacts verified; their arithmetic/compiler gains belong to those projects |
| 2026-10-14 | Tutti gio_uring; DOCA GPUNetIO TCP; GPU-Initiated Communication | Storage/LLM adjacency and strongest contrary evidence | Tutti is a GPU-side storage ring, rather than the same Linux TCP path; current DOCA and October2026 preprints make generic CPU-removal claims crowded |

## PDF corpus

No new PDFs were requested or downloaded. Primary paper links below were inspected directly; no local corpus or second citation database is created.

## Claim-oriented novelty map

| Plain claim | Closest prior work | Same-claim risk | Potential difference to prove | Baseline implication | Expansion opportunity |
|---|---|---|---|---|---|
| GPU code initiates I/O without a steady userspace polling proxy | LLVM GPU io_uring; GPUNetIO; upstream BPF loop | High | These primitives alone are not new | Cite enabling mechanisms; compare efficient CPU orchestration numerically only on matched work | Demonstrate complete application/connection/ownership behavior beyond an isolated operation |
| Existing computation can be compiled or executed as a persistent/fused GPU program | AMK, MPK and established CUDA Graph execution | High | Model arithmetic/compiler optimization is deliberately reused | Same backend/weights/precision on both transport paths; do not attribute backend gains to eBPF | Qualify different unmodified compute interfaces |
| Reused model computation serves ordinary Linux TCP/HTTP requests through eBPF admission and completion with no steady userspace request worker | GPUnet, LLVM GPU I/O, DOCA GPU services; adjacent Tutti | Medium, unresolved for the full combination | Kernel TCP semantics, programmable queue ownership, streaming reply lifetime and reusable backends may differentiate the integrated service | Efficient matched native TCP remains the main runnable competitor; hardware-specific/direct-NIC results are not interchangeable numbers | Broader supported backends, real remote traffic, offered load/tails and comprehensive CPU accounting |

The last row is a candidate difference, not a demonstrated absence of prior work or an established contribution. Preserve the full user target; this map neither replaces it with a print demo nor promotes a local timing win into a universal driver claim.

## Closest work

| Work | Verified claim/method and artifact | Evaluation relation | Difference or open boundary |
|---|---|---|---|
| [GPUnet, OSDI2014](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/kim), [paper](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-kim.pdf) | GPU socket APIs/network services; CPU I/O proxy coordinates NIC and host namespace | Accepted GPU-server precedent, latency/throughput/concurrency | Broad GPU-native service claim overlaps; its proxy architecture differs from the selected no-userspace-worker path |
| [LLVM GPU-driven I/O, August2025](https://blog.llvm.org/posts/2025-08-04-gpu-io-uring/), [artifact](https://github.com/phd-artifacts/gpu-libc-artifacts) | AMD fine-grained SVM exposes Linux SQ to device code; kernel polling replaces user RPC/polling for a print prototype | Direct same-mechanism feasibility precedent | Cannot claim first GPU-owned Linux ring or first removal of userspace polling. Console/file direction is not this complete TCP/LLM server contract |
| [Linux io_uring BPF operations](https://github.com/torvalds/linux/blob/v7.3-rc3/io_uring/bpf-ops.c), [loop](https://github.com/torvalds/linux/blob/v7.3-rc3/io_uring/loop.c) | struct_ops and kfuncs submit SQEs and coordinate CQ waits | Source-level enabling API, not an LLM service benchmark | bpf-gpu-server must contribute/verify its integration and lifetime policy; upstream loop APIs are not project novelty |
| [DOCA GPUNetIO3.5 architecture](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-architecture-and-design), [open artifact](https://github.com/NVIDIA-DOCA/gpunetio) | CPU setup followed by GPU/NIC data-path control; RDMA/Ethernet building blocks | Strong hardware-specific competing class | Direct NIC control already removes CPU critical-path mediation. It differs from retaining ordinary Linux TCP; runnable compatibility on this RTX5090 setting is not established |
| [Tutti, arXiv2605.03375v1](https://arxiv.org/html/2605.03375v1) | GPU-centric SSD KV object I/O, GPU-HBM SQ/CQ, layer-wise CPU preparation and resource-aware overlap | LLM storage/application adjacency; official numerical protocol differs | gio_uring mirrors the ring idea for storage rather than being the same Linux network ring. GPU compute/I/O contention is relevant; public runnable artifact and accepted venue are unverified |
| [GPU-initiated dSB, arXiv2610.06622v1](https://arxiv.org/html/2610.06622v1) | DOCA persistent receive/solve/reply service without a dedicated CPU data-path core | Recent same-solver socket comparisons, offered load and tail latency | Broad GPU service/control-removal claim overlaps. Its raw Ethernet custom EtherType protocol differs from Linux TCP; arXiv preprint, not verified accepted venue |
| [GPU-Initiated Communication, arXiv2610.01380v1](https://arxiv.org/html/2610.01380v1), [artifact](https://github.com/ParCoreLab/Dissecting-GPU-Communication-Experiments) | Minimal GPU versus tuned CPU-proxy RDMA transports distinguish issue/completion costs, load and resource footprint | Strong contrary evidence: GPU submission does not guarantee lower idle latency; queue/software/resource costs matter | Mechanism insights are relevant, while RDMA/datacenter hardware numbers cannot be substituted into this TCP/RTX5090 comparison |
| [AMK](https://github.com/RightNow-AI/AutoMegaKernel), [paper](https://arxiv.org/abs/2606.09682); [MPK](https://github.com/mirage-project/mirage), [OSDI2026 paper](https://www.usenix.org/conference/osdi26/presentation/cheng) | Existing model lowering/persistent-kernel computation assets | Compute-engine precedent, not matched transport superiority | Reuse rather than duplicate their optimization. Current AMK integration's stronger eager-HF failure remains failed; MPK compatibility here is untested |

The older DOCA2.8 [TCP/HTTP example](https://docs.nvidia.com/doca/archive/2-8-0/nvidia-doca-gpu-packet-processing-application-guide/index.html) explicitly uses CPU DPDK handling of TCP control packets and GPU data processing. That versioned example is not evidence that every current GPUNetIO path requires a CPU proxy. The October2026 dSB full text uses raw Ethernet frames; its socket baselines do not make its proposed service a Linux TCP implementation.

## Mandatory baselines

| Baseline | Artifact/version | Runnable status | Information and tuning | Protocol | Reproduction/fairness boundary | Consequence |
|---|---|---|---|---|---|---|
| Efficient native userspace TCP with identical reused model computation | Existing vllm_graph_user_network_wait at each frozen bpf-gpu-server source; common pinned runtime | Repeatedly runnable and measured; source/commands in findings | Same weights/KV precision/prompts/output budgets; current timed epoll and buffering policy | Source-native client TTFT/TPOT/token window and paired process blocks | Finite token-ID HTTP/local setting; not an official full text/API or SOTA server benchmark | A win prevents claiming native superiority; crossing-zero intervals prove neither equivalence nor noninferiority |

This is the current retained experiment's main competitor. Final admission for any new workload belongs to research-experiment-design. Component controls are not extra external baselines; adding every related system as a row would not strengthen a matched test.

## Experimental precedents and external assets

| Retained claim | Primary precedent/tool | Reusable design | Necessary difference |
|---|---|---|---|
| GPU-native network service | Accepted GPUnet OSDI2014 paper | End-to-end service correctness, latency/throughput and concurrent clients | Keep the same model computation and Linux TCP contract; do not reuse unrelated application numbers |
| Delivered LLM timing | [vLLM benchmark metric documentation](https://docs.vllm.ai/en/stable/benchmarking/cli/#understanding-the-latency-metrics) and actual vLLM computation | Client TTFT and per-request TPOT terminology, real reusable backend | Current finite cell window and token-ID frontend are custom glue, not full official benchmark equivalence |
| I/O control/compute tradeoff | Recent dSB and GPU-communication papers/artifacts above | Same-compute CPU alternatives, repeated load/tail tests, explicit resource and completion scope | They are recent preprints and different hardware/protocols; methods are transferable, numerical claims are not |

No formal paper RQ is invented. The current finite validation/protocol/lifecycle suites and paired analyses remain authoritative for their measured scope; external references do not repair missing evidence.

## Baseline candidates

| Candidate | Competing position | Reproduction risk/fairness | Current role |
|---|---|---|---|
| DOCA GPUNetIO or minimal direct-NIC transport | Bypass host submission/network stack for lower delivery cost | Requires compatible NIC/GPU/software and matched computation, reliability and arrival semantics | Citation-only competing class until a real compatible matched comparison is admitted |
| LLVM GPU io_uring prototype | Generic kernel-polled device I/O suffices without userspace RPC | AMD/SVM print artifact is a different platform/workload | Citation-only enabling/same-claim precedent |
| MPK/other official reusable model engine | Better existing computation rather than hand optimization | Architecture/API/checkpoint support on RTX5090 is untested | Potential compute asset, not another numerical transport baseline |

## Absorbable ideas

| Source | Idea | Implication and risk |
|---|---|---|
| LLVM and Linux BPF loop | Device-visible rings with ordinary kernel I/O | Reuse the mechanism; differentiate complete service ownership/lifecycle instead of claiming the primitive |
| GPU communication study | Separate operation decision, physical submission and completion; measure resource cost | Explain the actual CPU/kernel work left after worker removal. Do not call kernel-mediated TCP CPU-free or infer resource mediation from static counts alone |
| Tutti | I/O work can compete with model execution | Prefer reusable scheduling/interop mechanisms; source placement and model-resource evidence matter beyond simple path removal |
| GPUnet/dSB | Measure concurrent offered load and tails using unchanged computation | Strong future service evidence beyond current32-request closed-loop cells; no new infrastructure or workload is launched by this survey |
| AMK/MPK | Borrow compiler/runtime/model kernels | Preserve backend attribution and independent fidelity checks; the network driver does not inherit every upstream quality/performance claim |

## Adjacent communities

OSDI/NSDI/SIGCOMM: network services, transport semantics, programmable kernel paths.
ASPLOS/EuroSys/MLSys: GPU operating-system interaction, persistent compute, resource interference and model serving.
Storage/ICDE: GPU-initiated I/O and KV cache scheduling. Storage rings are adjacency, not interchangeable TCP baselines.

## Venue evaluation patterns

The accepted GPUnet precedent tests an application service and scaling, not only isolated launch cost. Recent papers add matched computation, offered-load/tail distributions, CPU-proxy operating-state controls and explicit completion/resource scope. These suggest stronger future experiments; they do not require blindly reproducing every hardware-specific configuration or adding performance settings.

## Must-read list

The primary links in Closest work cover the declared threats. The LLVM implemented prototype supersedes the earlier cancelled proposal. Recent GPU-communication and dSB preprints should be checked before claiming general CPU-removal novelty or universal performance gains. Tutti's artifact/venue and current MPK hardware compatibility remain unknown.

## Novelty verdict

Generic GPU I/O, removal of a userspace polling proxy and megakernel computation each have high overlap. The integrated ordinary-Linux-TCP/eBPF/reused-compute service is a potentially differentiated target whose broad evidence is still incomplete; this is an inference from the verified mechanisms, not a priority or first-system claim.

Preserve the original goal and model-independent driver direction. Use the efficient matched native path as the current numerical competitor, and cite broader direct-NIC/kernel-ring precedents honestly. Reuse existing computation rather than start a model-specific arithmetic campaign.

[Finding0055](findings/0055-current-gpu-io-arrangements.md) records the current arrangement boundary. [Finding0056](findings/0056-independent-gpu-queue-publication.md) completes that intervention: both primary improvements are inconclusive, and selected source restores original publication after fresh independent review. This survey changes neither its conditions nor its primary interpretation. After that decision, prioritize a distinct discriminator or stronger complete-service evidence; another unchanged timing matrix is not justified.

## Reusable backend compatibility refresh (2026-10-15 UTC)

This bounded refresh asks which unchanged computation assets could fit the
existing transport-independent driver on the actual RTX5090, rather than
claiming another model compiler or changing the serving objective. Search
branches are hardware/API compatibility, model breadth and baseline quality.
Exact additional searches:

- site.github.com/mirage-project/mirage megakernel RTX 5090 Blackwell sm120
- site.github.com/RightNow-AI/AutoMegaKernel RTX 5090 supported models
- persistent megakernel language model inference Blackwell RTX 5090 official github 2026
- GPU initiated Linux TCP server eBPF io_uring kernel loop
- site.github.com/mirage-project/mirage "sm_120"
- site.github.com/mirage-project/mirage "5090"
- site.usenix.org "MPK: A Compiler and Runtime"
- site.github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090 accuracy correctness benchmark

Primary source inspection,
not search snippets, supports the following distinctions.

| Asset | Verified primary evidence | Reuse decision and remaining uncertainty |
|---|---|---|
| [MPK, OSDI2026](https://www.usenix.org/conference/osdi26/presentation/cheng), [Qwen3 demo](https://github.com/mirage-project/mirage/blob/mpk/demo/qwen3/demo.py) | Accepted venue is now verified independently of its README. The official artifact exposes device tensors and discovers SM counts. Its [current shared-memory utility](https://github.com/mirage-project/mirage/blob/mpk/python/mirage/utils.py) handles compute capabilities80/86/89/90/100, without a120 case. The [fall roadmap](https://github.com/mirage-project/mirage/issues/773) names B200/B300 as the current platform. | A promising general computation asset, with actual RTX5090 build/run/correctness still unqualified. B200/B300 support does not establish consumer Blackwell support; absence of a helper branch alone is not a complete runtime incompatibility proof. Do not start a local arithmetic or porting campaign on this evidence. |
| [Current AMK README](https://github.com/RightNow-AI/AutoMegaKernel) | The documented current coverage is the HF Llama family on CUDA sm75..120. Its highlighted int8-versus-BF16 cuBLAS measurements change precision. | Keep finding0048's actual pinned FP16 negative result and eager-HF failure. New upstream claims do not retroactively repair them, establish arbitrary-model coverage, or form a matched transport baseline. |
| [AlpinDale Qwen megakernel](https://github.com/AlpinDale/qwen_megakernel), [README](https://github.com/AlpinDale/qwen_megakernel/blob/master/README.md) | An actual BF16 Qwen3-0.6B/RTX5090-specific asset; its table compares against PyTorch HF, and its stated hardware coverage is that GPU. | An optional fixed-model asset, not a generic model interface or evidence of superiority over optimized vLLM. GPU-driven graph interoperation and independent fidelity here remain untested. |
| [Another Qwen3/5090 reproduction](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090) | The published reproduction guide uses an enforce-eager vLLM baseline and a fixed model. | Useful lead, but disabling CUDA graphs changes the competitor relevant to this project. Do not import its speedup as a current bpf-gpu-server result or spend effort matching a deliberately weaker baseline. |

These sources do not change the novelty judgment or authorize a new GPU run
during the active adaptive-wait matrix. Reuse existing model engines, preserve
the same computation on both I/O paths, and require an actual hardware/API
preflight before integration. An unmeasured upstream speedup is not evidence
that an adapter would improve this service. Stronger complete-service offered
load/tail evidence and independently qualified backend breadth remain higher
value than a speculative model-specific kernel rewrite.
