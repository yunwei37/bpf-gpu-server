# Reusing upstream megakernel backends

Current integration status,2026-10-17UTC (internal evidence):
[opaque cuBLAS/cuFFT graphs](graph-backends.md) and
[HF/vLLM model adapters](architecture.md) are the qualified reuse boundaries.
[0080](findings/0080-resident-queue-relief.md) completes shared32-bank queue
relief;[0081](findings/0081-small-graph-network-cpu.md) closes the bounded
small-work transport experiment without a predeclared CPU/P99 advantage.
AMK/other candidates retain their recorded negative or partial qualification.
The source landscapes below are dated discovery/audit records,not freshly
verified upstream compatibility or instructions to reopen closed experiments.


bpf-gpu-server's research direction is to reuse upstream model computation and
measure the Linux TCP/HTTP + eBPF/kernel-to-GPU serving path. Model-specific
operator tuning is not evidence of an eBPF benefit. A useful comparison keeps
the backend revision, checkpoint, precision, KV layout and batching behavior
the same while changing the request/control path.

This landscape combines source audit with separately linked local correctness
qualifications. Source presence alone is not GPU or serving qualification;
performance claims require matched measurements. Observation: 2026-10-11T00:48:05.450904+00:00. Sources were inspected from the
original bpfusion-dev Workspace; the existing executor and unrelated WIP were
preserved.

## Candidate landscape

| Candidate | Reusable scope | Relevant limitation / current evidence |
|---|---|---|
| [Mirage / MPK](https://github.com/mirage-project/mirage) | Tensor-program compiler, device task scheduler, online request/completion rings; several model builders | Existing CPU ring producer/drainer remains part of upstream serving. Native SM120 compatibility probes pass after fixes, but complete Qwen3-0.6B outputs fail the unchanged HF oracle under parallel and serial submission. Registered model names are not independent correctness evidence. |
| [Plow](https://github.com/infervisor/plow) | HF importer, packet compiler, Lean ordering checks and SM120 interpreter; existing native Qwen harness | Native/Rust/Lean builds pass after compile/link compatibility fixes and existing GF2 selection. Four32-token memchecks pass, but only two sequences match the unchanged HF oracle. Decode-only prompt ingestion; no qualified prefill or resident network adapter. See [qualification](findings/0023-batch-execution-qualification.md#upstream-plow-sm120-build-and-mixed-numerical-qualification). |
| [model-as-a-kernel](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel) | Architecture interpreter; a finite prompt-plus-greedy-generation loop within one launch; single and batched decode interfaces | CUDA capability 12.0 is declared and the card reports an SM120 measurement. Native SM120 build passes with unchanged computation; seven of ten64-token SmolLM2 sequences match HF exactly. Not a qualified strict replacement, online queue or kernel-TX adapter. See source boundaries and qualification below. |
| [Luminal](https://github.com/luminal-ai/luminal), [compiler design](https://blog.luminal.com/p/compiling-models-to-megakernels) | General inference compiler and symbolic megakernel work queue | The current CUDA-lite path uses host-launched cached graphs; the historical compiler megakernel design is a separate source claim. Broad architecture expressiveness does not prove runnable whole-model online coverage. |
| [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) | HF Llama-family import and whole-forward synthesis, with RTX 5090 evidence | Current generate.py loops on the CPU, lowers/runs each step, synchronizes and reads argmax. Its reported INT8-versus-BF16 gains change precision; they cannot establish our path benefit. |
| [AlpinDale qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) | Existing BF16 Qwen3-0.6B kernel targeting RTX 5090 | Body/final norm and LM-head/argmax use separate kernels. Global mutable launch/synchronization state needs a concurrency audit; the bulk no-sync API is not evidence of resident online admission. Native SM120 build now passes, but all32 complete64-token sequences fail both independent HF references, including the predeclared checkpoint-theta correction. See [qualification](findings/0023-batch-execution-qualification.md#alpindale-rtx5090-qwen-qualification-complete-numerical-rejection). |
| [Lucebox megakernel](https://github.com/Luce-Org/lucebox/tree/main/optimizations/megakernel) | Qwen3.5-0.8B hybrid DeltaNet/Attention implementation; separate Blackwell NVFP4 path | Current kernel is model-specific and batch 1. Half-precision and NVFP4 paths are different compute conditions. The standalone Python decoder drives steps and reads each output on the host. |
| [Mini-vLLM](https://github.com/BoundlessWindMoon/minivllm) | Small Qwen3 whole-layer CUDA decode implementation; HTTP and continuous-batching paths also exist | The inspected megakernel wrapper synchronizes and reads output on CPU per step. Qwen3.5 source presence is not qualified megakernel support. README says MIT but the inspected tree has no license file. |
| [MegaQwen](https://github.com/Infatoshi/MegaQwen) | Consumer-GPU Qwen3-0.6B cooperative decode implementation and detailed optimization experiments | Static batch1/short-context scope; separate LM head and CPU generation/argmax. README says MIT but no license file was found. A compute reference, not a general resident serving runtime. |
| [HazyResearch Megakernels](https://github.com/HazyResearch/Megakernels) | ThunderKittens-based Llama whole-forward implementation | Published build choices are H100/B200; a 5090 port is not verified. |
| [Cohere megakernel](https://github.com/cohere-ai/cohere-megakernel), [serving design](https://cohere.com/blog/megakernels) | OpenAI-compatible serving and paged KV around a decode megakernel | Current research release targets North Mini Code on one H100, BF16, batches1-8. Prefill pauses decode. Python controls admission/prefill; a C++ loop updates positions, launches, samples and streams each decode step. |
| [TileRT](https://github.com/tile-ai/TileRT) | Tile runtime and model backends for large multi-GPU serving | Current supported wheels target 8 B200s or 8 MI350-class GPUs, with exact PyTorch ABI requirements; core backend is distributed as a binary. Not a demonstrated single-5090 source adapter. |
| [ARK](https://github.com/microsoft/ark) | GPU-driven computation/communication loop-kernel framework and Llama2 example | A relevant execution-model precedent; current model breadth and 5090 compatibility still need verification. |


| [NVIDIA low-latency LSTM](https://github.com/NVIDIA/dl-lowlat-infer) | Apache-2.0 streaming LSTM persistent kernels, reference and ping-pong measurements | Small/medium always-spinning modes retain CPU publication/output polling; large mode launches per iteration. A useful non-LLM compute candidate, not a qualified network adapter. |
| [Qwen3-TTS adapter](https://github.com/gauravlochab/qwen3-tts-megakernel) | Apache-2.0 example reusing AlpinDale's trunk for speech on RTX5090 | Batch1; host/PyTorch codec head and sampling. Hidden-state cosine is not exact closed-loop agreement. A reuse example, not a ready GPU-only speech server. |
| [Qwen3 RTX5090 reproduction](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090) | Dynamic-persistent CUDA/CuTeDSL decode paths and HF comparison scripts | No license found; test accepts partial-prefix matches. Eager-vLLM README gains are not matched serving-path evidence. |
| [MaruthiV megakernel](https://github.com/MaruthiV/megakernel) | Qwen3 BF16 whole-forward implementation targeting A100/H100/T4 | Single-token input interface; validation skips absent callbacks and generation hook takes only the last prompt token. No license or full-prompt correctness established. |
| [UCCL mKernel](https://github.com/uccl-project/mKernel) | MIT distributed compute/communication fusion for GEMM, MoE and attention | Components, not whole-model serving. Hopper default; Blackwell source targets SM103a/B300 and eight GPUs, not qualified SM120/RTX5090 support. |

[Event Tensor](https://arxiv.org/abs/2604.13327),
[Ada-MK](https://arxiv.org/abs/2605.11581), and
[ForgeMegakernel](https://arxiv.org/abs/2609.12379) are additional relevant
compiler/synthesis papers. This audit did not establish a complete, directly
usable public artifact for these candidates.
[FlashInfer MegaMoE](https://flashinfer.ai/2026/09/22/mega-moe.html) is a
MoE-layer compute/communication implementation rather than a complete
resident model-serving backend.

## Pinned source findings

### model-as-a-kernel

Audited Hub v1 revision:
[5858c6ddbe04c6b749e81daeb47bcd1857c1e618](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel/tree/5858c6ddbe04c6b749e81daeb47bcd1857c1e618).
The initial source audit read CUDA, importer, bindings and tests without a model
run. A later native qualification is recorded separately in
[the qualification finding](findings/0023-batch-execution-qualification.md#upstream-model-as-a-kernel-sm120-qualification).

- The Python generate() implementation at lines 929-961 calls mak_run_seq()
  once for prompt consumption plus generation. Its alternative uses the same
  program with individual launches. The CUDA mega_kernel() at lines 1483-1556
  has a finite position loop and returns when that generation completes.
  This removes inter-token host launches for that call; it does not implement
  a service resident across requests.
- The interpreter consumes an int64 [n_phases, 16] program containing device
  pointers and dimensions. MegaModel owns the weights and buffers. Those
  lifetimes must remain valid through inference and final output ownership.
- from_pretrained() at lines 591-592 rejects attention projection biases.
  bpf-gpu-server's current Qwen2.5 loader explicitly loads those biases. A direct
  swap of the existing checkpoint into this importer is therefore unsupported;
  changing the model requires its own independent reference.
- build.toml declares CUDA capabilities through 12.0. Grid sizing queries
  occupancy; the CUDA implementation uses a software global barrier between
  phases. bpf-gpu-server's existing eight independent graph banks must not simply
  be assumed safe or equivalent for this full-grid kernel. Concurrent launch
  residency and the alternative tensor-batched path need qualification.
- generate_batch() performs per-sequence prefill, reads first tokens on the
  host and then launches a batched decode. Its existence is not proof of
  GPU-only online batching or request admission.
- Output tokens are written into device buffers. The existing Python API
  transfers the result back to the host. It does not publish bpf-gpu-server's
  system-scope produced/DONE fields or implement TCP streaming/reclamation.
- In tests/test_mak.py, test_additional_model_coverage() for Qwen3/TinyLlama
  permits at least 85% argmax agreement across 48 fixed-history positions
  (lines 250-292), alongside logit/KV tolerance checks. Mode equivalence is
  checked separately. These tests do not establish token-exact HF greedy
  generation for those models. test_generate_closed_loop_and_speed() compares
  the backend's own launch modes; its eager speed threshold is not an eBPF
  comparison.

The current card is narrower than parts of the source: the importer includes
additional architecture/RoPE handling. Source presence alone is not measured
support, so that extra breadth is not promoted to a bpf-gpu-server coverage claim.

### Mirage / MPK

Audited upstream revision:
[f9eb70c254acefc9f3667b2a973d0dcf25471fce](https://github.com/mirage-project/mirage/tree/f9eb70c254acefc9f3667b2a973d0dcf25471fce).

- python/mirage/mpk/online_pinned_runtime.py exposes request/completion rings,
  ready flags, slot token inboxes and row ownership/release acknowledgement.
  Its submit(), completion-drain thread and output handling remain CPU work.
  These existing interfaces make it a promising transport/ownership adapter
  candidate, not an already CPU-free network service.
- python/mirage/utils.py's shared-memory capacity helper has branches through
  compute capability 100 and rejects other values. CMakeLists.txt line 28 lists
  native architectures 75/80/86/89/90. This identifies compatibility questions,
  not proof that every 5090 execution path fails: the call path and generated
  tasks must be checked before deciding how to build or adapt it.
- The Qwen3 builder registers several names, including a Qwen3.5 alias.
  A registry entry alone does not establish correctness for a different
  hybrid architecture.
- A native build and bounded SM120 Qwen3-0.6B qualification now exist.
  The temporary compatibility patch repairs attention metadata, shared-memory
  capacity and LM-head synchronization. Independent LM-head probes pass,
  but the complete model fails all12 reference prefixes under both parallel
  and serial submission, from the first output token. MPK is not a qualified
  model backend or kernel-queue integration. The [qualification finding](findings/0023-batch-execution-qualification.md#upstream-mpk-sm120-compatibility-and-terminal-numerical-rejection)
  links the exact patch, commands and complete positive/negative raw receipts.

### Additional source checks

A supplementary read-only search and source inspection on2026-10-11 expands
the candidate set; none of these source observations is a local model run.

- AlpinDale revision [5030e154d39ecd054df03eb4dd9c8aa8185414d1](https://github.com/AlpinDale/qwen_megakernel/tree/5030e154d39ecd054df03eb4dd9c8aa8185414d1)
  has an MIT license. csrc/kernel.cu keeps barrier/position/token state in
  process-global pointers. csrc/torch_bindings.cpp's generate_nosync()
  invokes the bulk launcher and synchronizes once at the end; this establishes
  a useful API boundary, not autonomous service across arriving requests.
- Lucebox revision [ae039d0acc2f03cd0e8cff33836ad05e4f64d310](https://github.com/Luce-Org/lucebox/tree/ae039d0acc2f03cd0e8cff33836ad05e4f64d310)
  is Apache-2.0. Its megakernel README documents the separate Blackwell
  NVFP4 path; model.py and model_nvfp4.py generate() call step() in a CPU
  loop and read the result with item(). The latter also exposes step_many();
  its existence is not qualification of an autonomous online request loop.
- Mini-vLLM revision [c5bde277518942bbebec0730f5b6684f396d7042](https://github.com/BoundlessWindMoon/minivllm/tree/c5bde277518942bbebec0730f5b6684f396d7042)
  and MegaQwen revision [0bc1fdedb82744481373000c5c7d3c58795c7a3a](https://github.com/Infatoshi/MegaQwen/tree/0bc1fdedb82744481373000c5c7d3c58795c7a3a)
  have public source but no license path in their inspected trees.
- [FlashFormer](https://arxiv.org/abs/2505.22758)'s author repository migrated
  to [cheetah-lang/flashformer](https://github.com/cheetah-lang/flashformer/tree/4768b7684ddf94d4845e5c36e2370a3f3c5bd112).
  Its47-file tree exposes operators, memory/synchronization primitives and
  component tests; a complete whole-model runner and license were not
  established. Do not count this as a runnable backend solely from the paper.
- [AutoMegaKernel's experiments](https://github.com/RightNow-AI/AutoMegaKernel/blob/main/EXPERIMENTS.md)
  describe10/10 supported configurations including three named real
  checkpoints, configuration sizes and a toy. This is not ten independently
  qualified real-model serving backends.

## Selection and comparison implications

MPK is the strongest lead for an existing online queue/runtime interface.
model-as-a-kernel exposes a reusable finite generation interface, but the local
unchanged-compute qualification now has mixed strict-reference results. Luminal remains a compiler candidate
whose current public runtime needs deeper inspection. The native 5090 Qwen
implementation is a useful narrower implementation reference.

These are investigation priorities, not a final backend selection. Qualify
checkpoint semantics and the actual launch/control interface before comparing
speed. Reuse upstream arithmetic and credit its compute performance. Necessary
adapters concern request publication, GPU control, streaming output and
lifetime/reclamation; model-specific arithmetic improvements are a separate
claim.

For the causal path comparison, keep model/backend, precision, KV and
batching equal. A credible userspace alternative can submit an entire GPU
generation per request; forcing it to execute a slow Python token loop would
confound the effect. Existing serial host-control results do not isolate the
current concurrent graph path. Measure client TTFT/TPOT/completion and host
cost with explicit attribution and report shared-GPU conditions.

## Closest serving claim

[Blink](https://arxiv.org/abs/2604.07609) already removes the steady host CPU
from serving by combining a SmartNIC/RDMA request path with GPU-resident
scheduling, batching and KV management. "First CPU-free LLM serving" is not a
defensible bpf-gpu-server novelty claim from the current evidence.

bpf-gpu-server's candidate distinction is ordinary Linux TCP/HTTP plus
eBPF/sockmap and a kernel/GPU shared queue, with the existing Linux networking
and asynchronous kernel TX semantics. Kernel networking, eBPF and kernel TX
still use CPU. Removing the userspace request worker is not removing all host
CPU work; total cost and useful latency/throughput gains remain to be measured.


[NVIDIA's DOCA GPU packet-processing application](https://docs.nvidia.com/doca/sdk/doca-gpu-packet-processing-application-guide/)
already demonstrates GPU processing and replying to HTTP. CPU DPDK handles
SYN/FIN/RST and installs flow rules; CUDA kernels handle data and responses.
It bypasses OS sockets and kernel networking. Therefore GPU-handled HTTP itself
is not the distinction from this prior artifact.

[Linux device-memory TCP](https://kernel.org/doc/html/latest/networking/devmem.html)
already provides direct NIC/device-memory payload transfer while keeping TCP
header processing in the kernel. Its documented API retains userspace recvmsg,
sendmsg and buffer-return/completion handling. Zero-copy transfer and removing
the steady userspace request/control worker are different properties.

[Wave Persistent Kernel](https://dotwave.ai/technical-notes/wpk-persistent-kernel/)
is a first-party continuous-voice execution precedent: the resident program
advances a static cohort, while the host publishes input and waits for output.
Its benefit includes shared-weight batching, not solely eliminating dispatch.
No reusable public source artifact was established in this search; treat it as
a mechanism/evaluation reference rather than a ready backend.


## Further search and source checks

Expanded observation: 2026-10-11T05:53:47UTC onwards. This round used24 additional
queries across reusable/multi-model persistent runtimes, consumer-GPU kernels,
GPU-initiated serving, eBPF/sockmap, kernel-to-GPU inference and citation
branches from Blink. Search results were leads; judgments below use first-party
source, official proceedings or papers. No new model benchmark or clone was run.

Pinned source observations:

- NVIDIA/dl-lowlat-infer
  [a718567b47e8d80ac38d58ac76df8d98a0eaab3d](https://github.com/NVIDIA/dl-lowlat-infer/tree/a718567b47e8d80ac38d58ac76df8d98a0eaab3d):
  docs/persistent_timing.md distinguishes small/medium always-spinning and large
  launch-per-iteration modes. ModelPersistentBase.cuh lines204-220 publish CPU
  input signals and poll output. ModelAsSingleStepGemv.cu retains precompute,
  synchronization and an optional host preprocessing thread. README exposes
  three sizes, reference/ping-pong measurement and a CUDA120a build example.
  This is a useful short-compute/non-LLM lead without new model arithmetic;
  its persistent final timestep is not an entirely host-free whole pipeline.
- Qwen3-TTS
  [61cb00e0b1e50d058a5f3c67de1e2f99b845ce87](https://github.com/gauravlochab/qwen3-tts-megakernel/tree/61cb00e0b1e50d058a5f3c67de1e2f99b845ce87):
  README describes embedding injection, a changed host RoPE table and PyTorch
  codec-head/sampling. talker/megakernel_talker.py asserts batch1;
  validate_talker_trunk.py checks hidden-state cosine over110 prefill positions,
  not exact closed-loop outputs. Absorb the adapter pattern without treating
  its quality/performance claims as our qualification or importing its separate
  math optimizations.
- RTX5090 reproduction
  [5bb05f54c1339a8d1ccc8968ca2278772f38b120](https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090/tree/5bb05f54c1339a8d1ccc8968ca2278772f38b120):
  megakernel_dynamic/test_dps.py lines138-139 allow partial-prefix matches unless
  shorter than roughly half the compared sequence. This is weaker than our
  unchanged exact oracle. No license was found in the inspected tree/API.
- MaruthiV
  [f2c8fd5c7e221ffabd017dc85afbf4a133c0e12c](https://github.com/MaruthiV/megakernel/tree/f2c8fd5c7e221ffabd017dc85afbf4a133c0e12c):
  src/megakernel.cu lines86-106 accept one input_token/current_pos.
  host/validate.py lines142-161 skip absent callbacks, and the generation hook
  receives prompt_ids[-1]. A validation command is not complete-prompt proof.
  No license was found in the inspected tree/API.
- mKernel
  [97e916b3a731689e4ea98db60bfe5f8adcaf685a](https://github.com/uccl-project/mKernel/tree/97e916b3a731689e4ea98db60bfe5f8adcaf685a):
  Makefile lines34-48 already contain Blackwell SM103a/B300 despite the README
  roadmap. Do not call all Blackwell support absent or equate that component
  implementation with RTX5090 or a complete model server.

### Compute reuse and closest-work implications

[Blink sections4.2-4.3](https://arxiv.org/html/2604.07609v1#S4.SS2) reuse captured
TensorRT engines plus a persistent device scheduler. A whole-model megakernel
is not required. Its same-engine TensorRT-LLM comparison is a causal precedent.
This supports investigating mature graph-capturable compute alongside
megakernels. It does not establish that any existing framework graph is directly
reusable: [CUDA device-graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-requirements)
require a single device, supported node types, host instantiation/upload and
proper executable-graph lifetimes. CPU callbacks, dynamic allocation and
host-only sampling/control need inspection in the actual backend.

- [Lynx, ASPLOS2020](https://doi.org/10.1145/3373376.3378528), verified through
  [the authors' institution](https://cris.technion.ac.il/en/publications/lynx-a-smartnic-driven-accelerator-centric-architecture-for-netwo-2/),
  offloads server data/control to SmartNICs and demonstrates neural inference
  without host CPU involvement. Accelerator network services without host CPU
  are older precedents than Blink.
- [SplitRPC, SIGMETRICS/POMACS2023](https://sigmetrics.hosting.acm.org/opentoc/acm23_vol7_issue2_toc.html)
  optimizes accelerator RPC on commodity NICs while retaining CPU orchestration
  across runtimes. "No expensive SmartNIC" alone is insufficient differentiation:
  conventional Linux TCP and steady userspace-worker removal must be demonstrated
  together. A runnable official artifact was not established here.
- [LAKE, ASPLOS2023](https://utns.cs.utexas.edu/assets/papers/lake_camera_ready.pdf)
  exposes accelerators to kernel applications with
  [public source](https://github.com/utcs-scea/LAKE). Its artifact uses a userspace
  API-remoting daemon. Kernel access to GPU compute alone is not our distinction;
  avoiding the steady request/control upcall is the candidate boundary.
- [Concordia](https://arxiv.org/html/2606.23521v1) is a persistent-queue/checkpoint
  precedent. Its paper bounds main claims to checkpoint/recovery and warns
  against interpreting executor microbenchmarks as faster static-graph decode.
  A reusable official artifact was not established in this search.

Investigation priorities remain backend correctness, actual request/step
control, online admission, KV/output ownership, streaming/reclamation and model
coverage. One compiler name or more Qwen forks do not prove broad support.
A qualified second architecture or short-compute model is more informative
than another tuning campaign on the same checkpoint. Credible optimized
userspace controls, CPU-contention/tail behavior and attributed costs are
proposed experiments, not established performance claims or a replacement of
the user's goal.

## Additional graph-runtime and networking artifact search

Read-only observation: 2026-10-11T06:05-06:10UTC. Queries covered multi-model
persistent inference, device-side task graphs, GPU-initiated RPC, Linux/eBPF
and socket paths, ForgeMegakernel artifacts, DITRON, Mustard, Paella, GPUnet
and GPU-Ether. An awesome list supplied leads; the decisions below use pinned
author source, primary papers and official proceedings. No new backend was
installed or run in this search.

- [DITRON, ICML2026](https://proceedings.mlr.press/v306/zheng26ai.html) has an
  actual public compute artifact in
  [Triton-distributed revision7908e4ea9010bb2238f9f05b28c904fd4b71c60a](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a).
  Its MIT-licensed source includes ModelBuilder, DenseModel, paged KV, a Qwen3
  chat server and a decode benchmark. The megakernel guide reports BF16,
  batch1/context512 on8 H800 or8 H20 GPUs, including a CUDA-graph baseline.
  dense.py lines191-200 runs embedding/copy outside the generated kernel;
  model_server.py lines75-88 retains CPU generation and per-token .cpu().item().
  This is a concrete compiler/backend candidate, not an autonomous server or
  measured single-5090/multiple-architecture qualification.
- [Mustard, ICS2025](https://doi.org/10.1145/3721145.3730426) has MIT source at
  [38b0e5955829e12a00acc5974776ccbff6ea8bb9](https://github.com/ParCoreLab/mustard/tree/38b0e5955829e12a00acc5974776ccbff6ea8bb9).
  mustard/mustard.h lines108-129 has a GPU scheduler that dequeues ready
  subgraphs and invokes cudaGraphLaunch(..., cudaStreamGraphFireAndForget).
  Existing runnable workloads are LU and Cholesky; CUDA12.3/NVSHMEM/MPI are
  dependencies. This supplies reusable scheduling code and a compute-preserving
  precedent, not an HF importer or ready LLM network server.
- [Toward GPU-centric Networking on Commodity Hardware, EdgeSys2024](https://dejankosticgithub.github.io/documents/publications/gpucentric-edgesys24.pdf)
  already evaluates GPU-driven inference with RDMA on ordinary Mellanox NICs,
  including TVM-compiled SqueezeNet and superresolution. Public
  [apps revisionb7ee60ea8bb67837d5ae6753cec2449ef34af715](https://github.com/rdma-from-gpu/apps/tree/b7ee60ea8bb67837d5ae6753cec2449ef34af715)
  has TVM loading in executor.cu lines379-409 and self tail-launch in
  lines355-376. It retains initialization/connection management but supports
  sleeping host control during serving. It uses RDMA/peermem/PCIe-P2P, rather
  than ordinary Linux TCP. No LICENSE file appears in this apps tree;
  reuse licensing across its related repositories remains unestablished.
  Both CPU-free inference and no-SmartNIC operation have older precedents.
- [GPUnet, OSDI2014](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/kim)
  already exposes GPU socket APIs and a face-verification service.
  [The authors' implementation account](https://acsl.ece.technion.ac.il/gpunet-networking-abstractions-for-gpu-programs/)
  explicitly retains a CPU NIC-control proxy; TCP also requires CPU transport
  and data movement. GPU socket syntax is therefore not equivalent to removing
  a steady userspace worker.
- [GPU-Ether, INFOCOM2021](https://netsys.ajou.ac.kr/publication/jung2021gpu/jung2021gpu.pdf)
  already implements persistent GPU packet I/O on commodity Intel Ethernet.
  It maps descriptors/doorbells using driver changes and GPU P2P.
  This is a packet-I/O precedent, not evidence of Linux TCP/HTTP compatibility
  or a complete LLM serving backend. An ordinary Ethernet NIC alone cannot
  distinguish bpf-gpu-server.
- [Paella's SOSP2023 artifact](https://github.com/eniac/paella/blob/master/sosp23_artifact/README.md)
  supplies public model-serving traces and direct-CUDA/Triton comparisons.
  It represents sophisticated userspace/software GPU scheduling. It is an
  evaluation/baseline lead, not a drop-in modern SM120 compute backend; its
  artifact uses older T4/P100 hardware and requires reproduction work.
- [ForgeMegakernel](https://arxiv.org/html/2609.12379v1) reports fourteen
  generated decode tasks across eight models and integration into SGLang,
  with independent intermediate-state checks. This search still finds no
  author-linked complete public repository. The model-generation campaign
  is a separate compute contribution; adopting its reported gains would not
  measure an eBPF benefit.

Together with Blink, these sources favor retaining two compute-reuse options:
upstream megakernel execution and device-launched captured computation.
Choosing the latter does not inherently require inventing model arithmetic.
Actual graph eligibility, full prompt/decode correctness, sampling, dynamic
shapes and buffer lifetimes still need backend-specific verification.

The proposed research boundary remains Linux TCP/HTTP + eBPF/kernel queue +
resident GPU control + asynchronous kernel TX, with demonstrated streaming and
recycling. Linux/eBPF still consume CPU. Same-model/same-precision/same-batching
userspace controls, total attributed host cost, CPU contention and client tail
latency are necessary evidence; merely eliminating launches or using a normal
NIC is insufficient novelty. More model-specific Qwen forks, new quantization
and repeated numerical tuning are not needed to establish this path claim.

## Further public source leads and current graph reuse

Source observation: 2026-10-11UTC. Additional searches covered
whole-model Llama kernels, general persistent operator runtimes, non-LLM
model graph exports, Event Tensor/TIRx integration and dynamic megakernel
artifacts. These are source leads, not installed or GPU-qualified backends.

- [FlashFormer](https://github.com/cheetah-lang/flashformer/tree/4768b7684ddf94d4845e5c36e2370a3f3c5bd112)
  has public kernel primitives and attention/GLU tests. Its README describes
  a whole-model forward, but this tree does not establish a ready HF import
  or online-generation entry point. setup.py pins Torch 2.5.1 and depends on
  Cheetah; no LICENSE file or repository license metadata was found. Keep
  it as a Llama compute lead rather than assuming a drop-in multi-model server.
- [FlashRT](https://github.com/flashrt-project/FlashRT/tree/66b18eb766e73e2fcad406bc622bd3bfd70664c1)
  is Apache-2.0 and exposes model graph, buffer, stage and lifetime contracts
  in docs/model_runtime_api.md. This is a useful non-LLM reuse lead, with
  Pi0.5/VLA graph producers and SM120 source. Its API still describes host
  stage scheduling; opaque stages are synchronous and do not promise capture.
  docs/adding_new_model.md explicitly requires translating model forwards
  into kernel calls. Adapting an existing graph producer differs from
  implementing that model translation ourselves; device-launch eligibility
  and unchanged-reference correctness remain unverified.
- [GPUOS](https://github.com/Multi-V-VM/GPUOS/tree/b51648e3c1929f35b0afc790b96be011354df09e)
  is MIT and supplies persistent workers, host-mapped completion and runtime
  operator injection. The public README and scheduler implement host-side
  aggregation/publication of small tensor operations. This is a general
  execution-mechanism precedent, not evidence of a complete HF importer or
  autonomous TCP model service. The
  [paper](https://arxiv.org/html/2604.17861v1) also explicitly describes a
  host-managed work queue.
- [TVM's official TIRx account](https://tvm.incubator.apache.org/2026/06/22/tirx)
  explicitly says TIRx is not a complete megakernel compiler, and describes
  Event Tensor integration as follow-up work. The
  [Event Tensor paper](https://arxiv.org/html/2604.13327v1) remains a dynamic
  shape/dependency precedent; this search does not establish a complete,
  directly reusable released compiler artifact. Secondary summaries claiming
  a ready et-compiler are not used as artifact evidence.

The existing HF graph interface now separately qualifies Qwen2.5-0.5B and
SmolLM2-135M (Llama), each under resident and host-control dispatch, while
reusing upstream computation. See
[the final four-case qualification](findings/0023-batch-execution-qualification.md#hf-graph-reuse-across-qwen-and-llama-architectures).
This supports retaining a compute-preserving graph route alongside upstream
megakernels. It does not qualify the newly searched projects or resolve the
earlier MPK/Plow/MAK numerical failures.

## Expanded backend and counterexample search

Read-only source observation: 2026-10-11T10:33-10:38UTC. This follow-up
refreshes earlier leads and adds general compiled-compute alternatives.
No new backend was installed or GPU-qualified. Pinned revisions distinguish
source evidence from project performance claims.

- [Luminal revision77d3060](https://github.com/luminal-ai/luminal/tree/77d3060f01dc9d6f649687cf16a651e3f288aa0c)
  is Apache-2.0. Its January2026 author article describes compiler-generated
  symbolic megakernel queues, but the inspected current CUDA-lite executor
  is graph-based: device.rs line1097 instantiates a graph and line1340
  launches it from the host. runtime.rs exposes borrowed-stream/external-buffer
  interfaces and the model zoo includes Llama implementations. Do not transfer
  the historical megakernel description into a claim that this current path
  already provides persistent online model control.
- [MLC-LLM revision8978ea9](https://github.com/mlc-ai/mlc-llm/tree/8978ea9aec626d64096aaa92fbc89bda2e3ea802)
  is Apache-2.0 and contains actual Llama, Qwen, Gemma and Mistral model
  implementations plus a cached CUDA-graph allocation pass. C++ batch_decode
  retains running-request, KV/preemption and token-commit/sampling work on
  the CPU. This is a broad compiled-computation candidate; it is not qualified
  device-autonomous serving.
- [llama.cpp revision23b0202](https://github.com/ggml-org/llama.cpp/tree/23b0202a189c44a54625aadcb37a946dd1d6278d)
  is MIT. ggml-cuda.cu lines4603/4609 instantiate with flags0 and launch on a
  host stream; graph update likewise uses flags0. The current executable
  therefore cannot simply be handed to a device scheduler. Quantization,
  CPU offload and sampling must be equal across a future transport pair.
- [nano-vLLM revisionbb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)
  is MIT and offers a small adapter surface, but the inspected model runner
  directly constructs Qwen3. Capture covers the model forward while
  compute_logits, sampling and token-list transfer remain outside; the
  engine keeps CPU scheduling. Small source size is not broad model support.
- [MegaBake revision5a18b82](https://github.com/megabake/megabake/tree/5a18b8226ac39a35c91492825310dc2679ce83a1)
  is MIT and compiles supported torch.export graphs into cooperative task
  schedules. Unsupported operators can fall back to eager execution; the
  loader still binds pointers, resets dependencies and host-launches.
  A single-token Smol forward demonstration does not qualify full generation.
- [Fleet revision0275d46](https://github.com/ROCm/fleet-chiplet-megakernel/tree/0275d466c89bc3158e19118b62ad9b02d7f2a364)
  now has official Apache-2.0 source on its amd_mi350 branch. This corrects
  the earlier no-public-artifact observation. The MPK-derived chiplet scheduler
  and Qwen3-8B example target MI350, not the present RTX5090.
- [Cohere revision67d0b9c](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c)
  has Apache-2.0 serving source. runtime.cu's DecodeService loop still
  synchronizes each step, and launch.cuh line165 host-launches the megakernel.
  It is a useful strong-serving control design, with model/H100 limits above.
- [AlpinDale revision5030e1](https://github.com/AlpinDale/qwen_megakernel/tree/5030e154d39ecd054df03eb4dd9c8aa8185414d1)
  retains the separate body and LM-head kernels. Its generate_nosync API
  feeds tokens back on device, but kernel.cu lines1582-1603 are a host loop
  submitting body, LM head and update kernels per token. Removing per-token
  output reads does not remove per-token host launches.
- [TileRT revision0ea1937](https://github.com/tile-ai/TileRT/tree/0ea19371f120977761bfa0cf6a2d08415d8fce0e)
  refreshes the hardware boundary: release0.1.6 adds public GLM5.2/5.3
  ROCm wheels for MI350X/MI355X. CUDA models still target eight B200s and
  the core backend is binary-distributed. This is broader than NVIDIA-only
  coverage, but does not establish a single-SM120 source adapter.

Two first-party sources oppose treating CPU removal as an automatic speed win:
[NanoFlow](https://github.com/efeslab/Nanoflow) overlaps CPU batching/KV work
with GPU execution, and [CoPilotIO, OSDI2026](https://www.usenix.org/conference/osdi26/presentation/chen-guanyi)
moves GPU-initiated storage completion polling to CPUs to free GPU resources.
The latter is a storage result, not a TCP inference baseline. These motivate
efficient native userspace controls and measuring GPU polling costs.

[Current TensorRT-LLM migration documentation](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)
also corrects a version-sensitive reuse assumption: release1.2 removed its
TensorRT engine backend and retains the PyTorch backend. Blink's historical
TensorRT-engine reuse remains a design precedent, not a current drop-in recipe.
[AWS Transformer TKG](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/transformer-tkg.html)
is configurable multi-layer compute on Trainium2/3, not CUDA or a complete
networked generation loop.

The candidate contribution remains a Linux TCP/eBPF integration that removes
the steady userspace request/control worker while retaining reusable GPU
computation and correct streaming/reclamation. Blink already demonstrates
host-CPU-free LLM serving and graph reuse; neither property alone is new.
Linux TCP/eBPF and asynchronous kernel TX still use host CPU. Backend-independent
interfaces and improved attributed cost/tail behavior are hypotheses requiring
matched evidence. Reusing a backend does not imply arbitrary-model support.

For implementation selection, favor the existing qualified HF graph interface
and general compiler/graph interfaces over another bespoke Qwen arithmetic
campaign. Keep MPK/Plow/MAK terminal qualification failures visible; resolve
upstream interface/correctness compatibility only where it enables that serving
question. Model-specific, other-hardware and layer-only kernels remain
references rather than an expanding implementation obligation.

General-model graph reuse is another candidate family:
[ONNX Runtime's CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
supports captured graphs with GPU I/O binding, but requires CUDA-only tensor
nodes and fixed shapes/addresses per graph ID; If/Loop/Scan and concurrent
Run calls on one captured session are unsupported. This API does not promise
a device-launchable executable graph.
[TensorRT enqueueV3 capture](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/performance/optimization.html)
supports models without mid-pipeline CPU interaction, and
[TensorRT for RTX](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-cuda-graphs.html)
documents explicit graph-capture failure/fallback cases. These offer reusable
CNN/encoder or other short-forward computation without hand-written model
arithmetic. Actual device-graph node eligibility and executable/buffer lifetime
would still need qualification; ordinary host graph replay is insufficient.

## Additional execution-interface checks

Read-only source checks on2026-10-11 extend the candidate survey without
changing the measured model/dispatcher or installing another backend.

[ARK revision3e8539d](https://github.com/microsoft/ark/tree/3e8539d2f21365e1a3cc423fdbac35035fe2cedd)
exposes caller tensor mappings at launch, which is a potentially useful
compute-reuse interface. Its loop executor launches ark_loop_kernel once,
but Executor::run writes an iteration count to a mapped host flag and
Executor::wait polls that flag
([executor.cpp lines944-1009](https://github.com/microsoft/ark/blob/3e8539d2f21365e1a3cc423fdbac35035fe2cedd/ark/api/executor.cpp#L944)).
The public Llama2 generator still copies input IDs, calls runtime.run,
reads logits and performs NumPy argmax inside a host generation loop
([generator.py lines142-147](https://github.com/microsoft/ark/blob/3e8539d2f21365e1a3cc423fdbac35035fe2cedd/examples/llama/generator.py#L142)).
GPU-driven operator execution does not establish a CPU-free online
generation/admission interface. Reusing this runtime would require control
adaptation; SM120 and complete-model correctness remain unqualified.

[CUDA PDL reconstruction](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction)
is a useful alternative-mechanism reference: its authors report97.4% of
matched persistent-megakernel throughput with an81-kernel graph, on H100
for Llama3.2-1B at batch1/P32/D128. This is their result, not a bpf-gpu-server
measurement. The
[experiment guide](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/blob/main/experiments/pdl-reconstruction/README.md)
explicitly permits free-running trajectory token mismatches and uses
fixed-position checks; that does not satisfy bpf-gpu-server's unchanged strict
greedy-token oracle. It supplies neither ordinary-Linux network ingress nor
resident online admission. CUDA PDL overlaps dependent kernel execution;
it does not itself remove host graph submission.
[Official PDL semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
and [device graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-requirements)
must be checked separately. This motivates preserving upstream kernels
behind a device-launchable graph rather than requiring one fused model kernel.

Two adjacent precedents constrain claim language without adding runnable
model backends. [NICA, ATC2019](https://www.usenix.org/conference/atc19/presentation/eran)
associates sockets with inline FPGA SmartNIC computation and includes
socket-level lifecycle/state integration. It does not establish commodity
Linux/eBPF-to-GPU LLM serving, but socket-to-accelerator integration alone
is not a new abstraction.
[eBPFML, eBPF2025](https://www.cs.columbia.edu/~kkaffes/papers/empowering-ebpf25.pdf)
proposes CPU matrix-accelerated models for kernel decisions, rather than a
discrete-GPU network serving pipeline; the paper describes a design vision,
not a general upstream-ready model-serving artifact.

The current HF graph capture also has a concrete configuration boundary:
it derives KV head dimension as hidden_size/num_attention_heads.
The locally cached
[Qwen3-0.6B config atc1899de](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
specifies head_dim128, hidden_size1024 and16 heads, so that derivation yields
64 rather than128. Installed Transformers5.19.0 Qwen3Attention uses the
explicit config.head_dim for its projections and cache shape. Supporting this
checkpoint requires respecting the discovered model configuration, not
rewriting Qwen3 arithmetic. This is source/config evidence only: the measured
Smol/Qwen2 code remains frozen, and Qwen3 graph/network correctness has not
been qualified.

For a future strong external serving comparison, SGLang's native token-ID
interface avoids imposing tokenization/text work on only one system.
[Release0.5.21 tokenizer_manager](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/tokenizer_manager.py)
accepts input_ids when tokenizer initialization is skipped, maintains
state.output_ids, and returns cumulative or incremental output IDs according
to the streaming setting.
[BatchTokenIDOutput](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/io_struct.py#L1526)
explicitly identifies its output_ids path with skip_tokenizer_init.
This makes the official native HTTP API a stronger candidate than an
in-process eager-loop proxy. It still uses JSON/SSE rather than bpf-gpu-server's
binary body, so protocol/framing and model-computation differences need
explicit interpretation, independent output checking and native baseline
engagement before any numerical claim.

The owning bpf-gpu-server environment has neither SGLang nor vLLM installed.
The2026-10-11 release/PyPI read identifies SGLang0.5.21; its
[package metadata](https://pypi.org/pypi/sglang/0.5.21/json)
requires Torch2.13.0, whereas this Workspace has Torch2.14.1.
Reproduction would need a test-local dependency environment; do not replace
the measured environment or another owner's active scheduler. No installation,
server run or external-baseline performance evidence follows from this survey.

## Earlier GPU networking and evaluation precedents

Follow-up read-only search on2026-10-11T21:53UTC used ten materially different
queries on GPU sockets, event loops, GPU-initiated inference, persistent
operator runtimes and efficient CPU alternatives. The declared questions were:
whether GPU request ownership itself is new; whether device graphs can reuse
complete upstream models without a megakernel; and which CPU alternative makes
an efficiency claim meaningful. This branch adds networking/evaluation
precedents rather than another obligation to install model kernels.

- [GPUnet, OSDI2014](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/kim)
  already supplies GPU socket/stream abstractions and GPU-native services.
  The [authors' description](https://acsl.ece.technion.ac.il/gpunet-networking-abstractions-for-gpu-programs/)
  explicitly retains CPU assistance for TCP/non-RDMA transport.
  At [revision10a6da3](https://github.com/ut-osa/gpunet/tree/10a6da3b4bbfced885272c535ba661301f49a551),
  core/hostloop.cpp calls CPU recv/send at lines673/839 and performs
  CUDA staging copies on its TCP branch. GPU socket APIs alone therefore
  cannot distinguish bpf-gpu-server; eliminating this steady userspace proxy while
  keeping Linux transport is the relevant integration question.
- [GLoop's public artifact](https://github.com/CPFL/gloop/tree/25557429de5283a1803f201d8d054ffe5fc26109)
  cites its SoCC2017 publication. Its [earlier author paper](https://www.cs.utexas.edu/~mars2016/workshop-program/MaRS_2016_paper_6.pdf),
  sections3.2-4.2, uses host/device event loops and shared-memory RPC;
  the CPU performs requested I/O and publishes completions. Device polling
  and event-driven callbacks are established mechanisms, not new by themselves.
  This older Kepler/Pascal runtime is a citation/control-design precedent,
  not a locally qualified modern model backend.
- [Toward GPU-centric Networking on Commodity Hardware, EdgeSys2024](https://girondi.net/pub/gpunetworking24-cc.pdf)
  is particularly close: sections4-5 reuse TVM computation in a self-launching
  CUDA graph containing wait-input, compute, send-output and tail-launch.
  It evaluates GPU-driven RDMA inference and CPU consumption, on A100/L40.
  [The official artifact](https://zenodo.org/records/13332456) exposes the
  CUDA RDMA shim and model-serving applications.
  [Pinned apps revisionb7ee60e](https://github.com/rdma-from-gpu/apps/tree/b7ee60ea8bb67837d5ae6753cec2449ef34af715)
  confirms executor.cu's TVM graph loading and device tail launch at lines355-429;
  worker.cu lines659-665 supplies a sleeping host lifetime alternative.
  Reused model graphs plus CPU-free RDMA serving are already demonstrated.
  Ordinary Linux TCP/eBPF integration remains a candidate distinction;
  changing the transport must deliver measured practical value, not just
  replace an existing mechanism with a new label.
- [GASPP, ATC2014](https://www.usenix.org/conference/atc14/technical-sessions/presentation/vasiliadis)
  already offloads flow state and TCP stream reconstruction to GPUs.
  It addresses traffic-processing modules, not the current Linux socket
  service with reused autoregressive computation. Broad claims of first GPU
  TCP/stream processing are therefore untenable.
- [GPU-Initiated Networking for NCCL](https://arxiv.org/abs/2511.15076)
  explicitly distinguishes direct GPU-initiated and CPU-proxy plugin semantics.
  A device-side API name is not evidence that its selected implementation has
  no steady CPU proxy. This is a communication/interface precedent, not a
  token-ID HTTP numerical baseline.
- [Raising the Bar for Using GPUs in Software Packet Processing, NSDI2015](https://www.usenix.org/conference/nsdi15/technical-sessions/presentation/kalia)
  finds that suitably optimized CPU implementations can be more resource
  efficient than GPU versions of packet-processing work. It does not settle
  bpf-gpu-server inference, but strengthens the requirement for an efficient CPU
  control instead of a busy-loop-only comparison.

The GPUOS source/paper inspected in this branch is the same pinned revision
already recorded above; no second candidate or new qualification is counted.
Remaining uncertainties are numerical/backend correctness on SM120, actual
device-launch eligibility, total attributed CPU cost, physical-network behavior
and fair strong-serving comparisons. Search coverage now includes early socket
and event-loop systems, current GPU-network APIs, whole-model compilers,
graph reuse, conventional serving alternatives and CPU-efficiency counterexamples.
Further implementation selection should follow those concrete uncertainties;
model-specific arithmetic, a custom TCP stack and porting every candidate to
the present GPU are unnecessary to isolate the requested serving-path effect.

## Additional general compilers and new public backend leads

Read-only follow-up on2026-10-11 after the readiness comparison. Queries
included persistent megakernels beyond the existing Qwen implementations,
GPU-resident graph launch,non-LLM compiled models,consumer-GPU artifacts,
and eBPF/TCP serving. Search/awesome-list/Reddit results were leads only;
decisions use primary papers,official proceedings and pinned author source.
No candidate was installed,built or GPU-qualified in this search.

- [MonoNN,OSDI2024](https://www.usenix.org/conference/osdi24/presentation/zhuang)
  is a general compiler for static whole-network inference,with an
  [official Apache-2.0 artifact at2ff5b42](https://github.com/AlibabaResearch/mononn/tree/2ff5b42c0c0bf7046ce2496f2f580fcdd2468d45).
  README lines123-124 expose a BERT example; lines177-180 describe generated
  CUDA/PTX and direct use. This is a concrete non-autoregressive reuse lead
  for testing model breadth and short-compute request costs without writing
  model arithmetic. Its published dependencies are TensorFlow2.9.2,
  Python3.8,CUDA11.6 and T4/A100/A10,so present SM120 compatibility is
  unqualified. Finite static inference is not online resident admission.
- [Rammer,OSDI2020](https://www.usenix.org/conference/osdi20/presentation/ma)
  supplies task-based compile-time scheduling and inter/intra-operator
  co-scheduling. The [MIT NNFusion source at00c1317](https://github.com/microsoft/nnfusion/tree/00c13173a155556e50510575f6682a6a654cd585)
  links the paper and provides a frozen LSTM example and model export paths.
  It broadens reusable computation beyond one hand-written LLM. Neither
  the general compiler nor its paper establishes an already resident
  Linux-TCP request server or local SM120 qualification.
- [PipeThreader,OSDI2025](https://www.usenix.org/conference/osdi25/presentation/cheng)
  uses software-defined pipelining and a task graph over specialized GPU
  units; its official proceedings link [TileLang](https://github.com/tile-ai/tilelang)
  as code. Attention/Mamba2 examples concern computation scheduling rather
  than complete online model serving. Use it as a compiler/mechanism
  precedent; adopting it does not require a new operator-tuning campaign.
- [OpenJet's megakernel at2993f47](https://github.com/L-Forster/open-jet/tree/2993f47fd044adedb4bd4709d655239cc22b671c/megakernel)
  is newly observed public AGPL-3.0 source for Qwen3.8-27B Q4_K_M,
  advertising SM86/89/120 and single-conversation speculative decoding.
  The pinned README limits supported weights and reports approximate
  fidelity,not token-exact agreement. qwen38_mk.cu line2487 launches a
  cooperative speculative cycle; capi.cpp line95 exposes one host step,
  and server/mk_server.py lines114-119 call it and return host token IDs.
  It therefore offers a narrow compute/interface reference,not a
  multi-model resident online backend or proof of our eBPF benefit.
  Its reported3090 speedups and precision/speculation choices cannot be
  transferred to the current FP16 graph comparison.
- [Weave](https://arxiv.org/html/2609.21483v1) fuses MoE dispatch,expert GEMMs,
  activation and combine with dynamic compute/communication scheduling.
  It is a layer/distributed-compute mechanism,not complete prompt-to-token
  network serving. No complete author-linked serving artifact was
  established here; do not turn this citation into another required port.
- [GPU Resident Inference Lab at e27e56d](https://github.com/manishklach/gpu-resident-inference-lab/tree/e27e56d4d192d8e76c070af26f26dbfdd929c01c)
  explicitly describes deterministic placeholder transformer math,CUDA
  control-flow scaffolds and future real-model TTFT/ITL/tok/s. It is
  excluded as a ready inference backend despite matching search keywords.
  Its research-use license file is not an established permissive license.

This branch reinforces the existing two reuse options: upstream megakernels
and device-launched captured/compiled computation. Whole-model fusion,
GPU-side task queues and CPU-free serving already have prior work. The
candidate distinction remains correct Linux TCP/HTTP + eBPF/kernel/GPU
integration that eliminates a steady userspace request/control worker,
with measured practical value and explicitly attributed remaining CPU cost.
A compiler's model breadth is not bpf-gpu-server's tested model breadth.

Decision implications: retain the currently qualified graph backend for
matched path comparisons; treat MonoNN/NNFusion as general/non-LLM reuse
leads and OpenJet as a narrower consumer-card reference. No need to port
every candidate,write model-specific GEMM/attention,implement a TCP stack,
or inherit speculative/quantized compute gains into the eBPF claim.
Physical-network behavior,total attributed CPU and a strong official serving
baseline remain higher-value unresolved acceptance boundaries. The latest
[readiness result](findings/0028-socket-readiness.md) does not establish
an advantage over its efficient native control.


## Stock I/O interfaces and additional closest work, 2026-10-13

The [gpu-libc-artifacts prototype atba41b17](https://github.com/phd-artifacts/gpu-libc-artifacts/tree/ba41b17eeaf2bc0f0f04417404bb35f1005194b2)
and its [author's report](https://gist.github.com/rodrigo-ceccato/6475e8a3aaa669661386203be38b9093)
demonstrate GPU-written Linux io_uring submission entries with a kernel polling
thread for printing. The described HSA/SVM path is a useful reuse candidate,
and GPU-initiated ordinary kernel I/O is prior work. This is not a qualified
NVIDIA TCP/model server; the author lists NVIDIA evaluation as future work.
Do not equate a printing demo or a setup proposal with complete accept/recv/
send, idle wakeup, disconnect and buffer-lifetime support.

[Official liburing setup flags](https://github.com/axboe/liburing/blob/master/man/io_uring_setup_flags.7)
describe SQPOLL and caller-provided ring memory via NO_MMAP.
[The polling lifecycle](https://github.com/axboe/liburing/blob/master/man/io_uring_sqpoll.7)
requires explicit wakeup when NEED_WAKEUP is set. The concrete kernel UAPI,
memory registration and default idle behavior must be checked on the running
kernel, rather than inferred from a version number or a flag name. NVIDIA's
[550 release note](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-54-14/index.html)
records removal of importing IO_URING buffers for Linux6.8 compatibility.
That historical warning does not establish behavior of the current610 driver
or caller-allocated ring memory. A bounded real NVIDIA/kernel-ring/TCP check
is necessary before claiming the current module can be replaced. The later
[stock TCP diagnosis](findings/0037-stock-io-uring.md) establishes GPU queue
submission and repeated receive-completion wakeup on the original RTX5090,
with exact client bytes and EOF. The later [finite real-model integration](findings/0038-io-uring-sockmap-models.md)
connects unchanged eBPF admission and HF computation for both checkpoints,
with GPU-owned accept/receive/send/close and kernel-side eBPF reclamation.
The [streaming extension](findings/0039-io-uring-streaming.md) additionally
qualifies exact1/64-token outputs for both checkpoints and final-only release.
The [multi-connection implementation](findings/0040-io-uring-concurrent.md)
adds real-ring/namespace correlation and an optional normal build target,
qualified for both HF checkpoints through eight-client streaming and abort
lifecycle checks. Overload/extended-operation and matched performance/CPU
benefit remain open; the default module path is retained.

[Tutti](https://arxiv.org/html/2605.03375v1#S3.SS2) uses a GPU SQ/CQ design for
SSD-backed KV objects and integrates with vLLM. Its gio_uring mirrors the
asynchronous queue idea over GPU/NVMe infrastructure; it is not evidence that
stock Linux io_uring TCP operations work on this NVIDIA serving path. It also
reinforces that removing per-I/O CPU control and reusing inference computation
are existing ideas, rather than standalone novelty.

[GPU-Initiated Discrete Simulated Bifurcation](https://arxiv.org/html/2610.06622v1#S5)
(Oct5 submission) is a close mechanism comparison: resident receive/solve/reply
via DOCA GPUNetIO, contrasted with socket-based servers using the same solver
and client, including a persistent CPU proxy and load-dependent latency.
Its custom-EtherType/ConnectX path and dSB workload differ from Linux TCP,
eBPF parsing and reused LLM graphs. It rules out claiming resident GPU
request execution, a matched-compute CPU-proxy comparison or removal of a
dedicated userspace data-path core as first by themselves. The unresolved
bpf-gpu-server claim must identify the Linux/eBPF integration and its demonstrated
benefit or portability boundary; current null-inclusive transport results do
not supply a speedup claim.

The [optimized stock io_uring integration](findings/0041-vllm-io-uring.md) now
reuses the same transport header with the existing vLLM graph adapter. Both
architectures pass complete strict independent batch-history replay and the
shared-header HF lifecycle regressions. This expands the tested adapter
integration, not arbitrary backend support or measured performance/CPU benefit.

The following [matched optimized stock-TCP comparison](findings/0042-vllm-io-uring-performance.md)
now completes the local latency/throughput/known-worker CPU measurement. It
finds no reliable stock advantage, a negative Qwen long-throughput effect and
substantially higher measured worker CPU. The module-free implementation is
a qualified option, not evidence of CPU removal or broad serving superiority.
