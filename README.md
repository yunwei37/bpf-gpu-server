# bpf-gpu-server

原名 BPFusion。

See [name migration](#name-migration) for historical source and command names.

Experimental serving transport over ordinary Linux TCP, with eBPF admission
and caller-supplied GPU execution. Userspace initializes the service; the
kernel-owned path manages request/reply lifetimes around reused CUDA graphs.
The transport is separate from model arithmetic.

~~~text
TCP/HTTP -> eBPF framing/admission -> shared queue
         -> GPU backend execution -> completion -> Linux TCP reply
~~~

Userspace loads weights, captures graphs and initializes the service. On the
kernel-owned paths it does not accept each request, launch each model step or
send each reply after startup. Linux networking, kernel workers and GPU I/O
control still consume resources:this is not a CPU-free server.

**Status: working research prototype on a qualified Linux/GPU stack.**
Unchanged cuBLAS/cuFFT graphs and two real checkpoints through HF/upstream
vLLM graph adapters have finite numerical/lifecycle qualification.
A recent-kernel io_uring/eBPF path needs no custom TCP kernel module.
General eBPF latency/throughput superiority over efficient native TCP is
**not established**.

## Paper and write-ups

The four-page report [*Does Kernel-Owned Networking Make GPU Serving Cheaper?
Lessons from an eBPF-Driven GPU Server*](paper/bpf-gpu-server.pdf) summarizes the design, the
same-executor comparison and why the kernel-owned path saves no CPU: per-request
work moves from user to system time because the Linux TCP stack and GPU
completion coordination remain on both paths. Blog posts:
[English](https://eunomia.dev/blog/2026/10/17/bpf-gpu-server/) and
[中文](https://eunomia.dev/zh/blog/2026/10/17/bpf-gpu-server/).

The [paper artifact](ARTIFACT.md) maps every table, data figure and key number
to original data, with an offline replay and a bounded GPU smoke.

## Start here

For the model-independent surface, follow the [captured-graph runbook](docs/graph-backends.md).
It includes prerequisites, normal builds, cuBLAS/cuFFT examples, a binary client,
the same-executor native comparator and cleanup. With those dependencies active:

~~~sh
make build/server.bpf.o build/bpf_gpu_server_load \
  graph-server-kernel-network graph-server-user-network-wait
./build/bpf_gpu_server_load stream-uring-attach
python3 examples/torch_graph_server.py build/graph_server_kernel_network.so fft --seconds 120
~~~

Use the guide's client in another terminal. Input/output are fixed raw32-bit
words; graphs and buffers stay caller-owned. Stop/detach this service before
starting the native comparator. This is an experimental ABI, not a stable
arbitrary-kernel plugin.

For LLM generation, the separate
[Getting started](docs/getting-started.md) covers dependencies, model download,
build, start, client, native comparison and cleanup. It uses the explicit
module-free vLLM target; the legacy module transport remains separate.
[Fresh-environment validation](bench/results/onboarding_20261016/) passes both
models/transports with real HTTP/TCP requests. The
[post-rename validation](bench/results/rename_20261017/) records ordinary builds,
both models, cuBLAS/cuFFT and offline paper replay with the current names.

Tested stack:Linux 7.3-rc3, x86-64, RTX 5090(sm_120), CUDA toolkit 13.0,
Python 3.12 and the [pinned Python environment](requirements.txt). Ordinary builds
were qualified with system libbpf 1.3 headers and the 1.6.3 library; see the
[header/runtime compatibility note](ARTIFACT.md#live-reruns).
The BPF loop needs its recent io_uring BPF/BTF/UAPI interfaces.
Another GPU or simply a recent Linux version does not establish compatibility.
See the [host record](bench/results/HOST.md).

With the documented dependencies and environment active:

~~~sh
make build/server.bpf.o build/bpf_gpu_server_load \
  vllm-mixed-uring-loop vllm-mixed-user-network-wait
./build/bpf_gpu_server_load stream-uring-attach
./build/vllm_mixed_uring_loop Qwen/Qwen2.5-0.5B-Instruct 120
~~~

Wait for the ready line. In another terminal with the same Python environment:

~~~sh
python3 tools/llm_tcp_client.py --http --gen 32 --rounds 1
~~~

The client tokenizes text and decodes returned IDs. The server's HTTP body
contains binary token IDs, not JSON or an OpenAI API. The guide also covers
pinned model revisions, Smol and cleanup. The external-vLLM HTTP baseline
client is a different protocol.

## Supported scope

| Surface | Qualified scope |
|---|---|
| Models | Qwen2.5-0.5B-Instruct and SmolLM2-135M; Qwen/Llama architectures |
| Compute | Caller-supplied CUDA graphs; reused cuBLAS/cuFFT or HF/vLLM model computation |
| Network | Binary TCP and Content-Length HTTP POST; opaque words or streamed token IDs |
| GPU | Graph dispatch/output publication; generation adapter also owns prefill/decode/KV progress |
| Linux | Framing, admission, asynchronous TCP and reclamation |
| Module-free path | Explicit HF/vLLM io_uring BPF-loop targets on the tested stack |
| Internal comparator | Efficient native epoll TCP with the same resident GPU compute |
| Validation | Finite exact outputs/replay, fragmentation, concurrency, reuse, reject, half-close/reset/stop |

Default capacity is8; graph input/output counts are each1..64 words.
Generation uses the same framing for1..64token inputs/outputs.
Explicit16-bank vLLM qualification uses a matching shared-page ABI and separate
builds. See [architecture](docs/architecture.md) and
[capacity evidence](docs/findings/0064-batch-capacity.md).

For sustained mixed serving, the [qualified32-bank profile](docs/getting-started.md#qualified-32-bank-mixed-profile)
uses a separate matching build. The completed five-block comparison relieves
the shared 16-bank queue limit; see the results below. Legacy defaults remain
unchanged. This is a common capacity repair, not an intrinsic eBPF speed advantage.

An experimental [opaque CUDA Graph adapter](docs/graph-backends.md) also runs
unchanged cuBLAS/cuFFT computation through both transports. It accepts fixed
input/output buffers and is separate from LLM generation; arbitrary-CUDA
plugin stability and compatibility are not established.

## Results and boundaries

The internal comparator is efficient native epoll with the **same resident GPU
executor and protocol**. Native also uses GPU-controlled execution; there is no
Python per-token or per-request compute loop in this comparator. Official vLLM
is a separate external reference with its normal API, scheduler and KV cache.

[The completed 16/32/official comparison](docs/findings/0080-resident-queue-relief.md)
uses Qwen2.5-0.5B FP16 on RTX 5090, a remote client, 192 requests/s for 60 seconds,
five process blocks, short prompts and 64 output tokens:

| Path | Delivered tokens/s | Completion P99 | Owned group CPU/request |
|---|---:|---:|---:|
| Resident native, 16 slots | 11,640 | 3337 ms | 0.751 ms |
| Resident native, 32 slots | 12,265 | 217 ms | 0.523 ms |
| Official optimized vLLM | 12,249 | 406 ms | 8.327 ms |

All 172,800 requests complete with no observed failures. Increasing common
active capacity removes sustained queue growth; mean token time slightly
worsens. The fixed load is near its 12,288 token/s offered ceiling, so this does
not rank maximum capacity. Device-wide energy is higher than official vLLM;
shared host CPU, finite trajectory flags and client/protocol limits remain
explicit in the complete report. **These results do not measure an eBPF gain.**

Earlier [same-executor transport comparisons](docs/findings/0078-sustained-official-vllm.md)
establish no general kernel-path speed advantage. The
[ordinary kernel-network targets](docs/findings/0079-kernel-network-source-integration.md)
and [opaque CUDA Graph adapter](docs/graph-backends.md) provide the reusable
implementation boundary; cuBLAS/cuFFT numerical and lifecycle checks pass in
the qualified cases. They do not promise a stable arbitrary-kernel plugin.

The [bounded small-graph experiment](docs/findings/0081-small-graph-network-cpu.md)
is complete: 60 screening cells with 2.52 million measured requests, followed
by three new-seed, five-pair confirmations with 600,000 requests. No main
request fails. Matched cuBLAS/cuFFT work, load through saturation, owned-service
CPU masks and remote clients establish **no predeclared useful CPU/P99 advantage**.

| Confirmation at 1,000 requests/s | Group CPU K/N, t95 | P99 K/N, t95 |
|---|---:|---:|
| FFT, unrestricted | 1.011 [0.995, 1.027] | 1.254 [0.514, 3.058] |
| GEMM, unrestricted | 1.019 [0.967, 1.074] | 1.160 [0.497, 2.706] |
| GEMM, one logical CPU | 1.129 [0.956, 1.334] | 2.589 [1.196, 5.607] |

Ratios below 1 favor K for CPU/P99. These are nominal five-pair intervals;
crossing 1 does not establish equivalence. Single-CPU P99 is adverse.
Userspace CPU falls while system CPU offsets it; whole-host savings remain
unestablished. Full tails, observer cost, client timing faults and actual
cleanup are explicit in the report.

The bounded performance search is closed. bpf-gpu-server remains an OSS research
prototype for the qualified transport/graph boundary, with no general speed,
total-CPU or stable arbitrary-kernel-plugin claim.
[Daily progress and decisions](docs/progress.md) retain repository upkeep and
the reasoning behind the next substantive action.

[Evaluation](docs/evaluation.md), [all findings](docs/findings/) and their raw
archives preserve the full history, including failed and unselected candidates.
The [related-work map](docs/background-related-work.md) and
[backend audit](docs/upstream-backends.md) distinguish existing GPU networking,
GPU-initiated I/O and megakernel computation from the integration question.

## Supported limits and remaining work

This is a research prototype on the qualified stack, with short fixed-shape
requests and finite correctness/lifecycle evidence. Longer reset/churn loads,
broader CUDA/library and kernel/hardware compatibility, text/JSON APIs, TLS,
long context and multi-GPU serving remain unqualified. The LLM scheduler is a
separate adapter; the graph interface takes fixed buffers of at most 64 words.

Process-group, cgroup, host and device totals overlap. Kernel workers and
IRQ/softirq still consume resources; lower userspace CPU alone is not proof of
lower total service cost. Do not sum these scopes or call the service CPU-free.
The historical 6.36 ms token-gap difference remains unreproduced/unallocated.

These boundaries do not justify repeating closed micro-optimizations or
rewriting model-specific attention/matrix kernels.

## Repository map

| Path | Purpose |
|---|---|
| bpf/ | Framing, admission, sockmap and io_uring BPF coordination |
| bpf/include/bpf_gpu_server_queue.h | Shared kernel/GPU ABI |
| executor/ | Reused compute adapters, resident dispatch and shared transport |
| tools/llm_tcp_client.py | Text-to-token TCP/HTTP demo |
| tools/bpf_gpu_server_load.c | Attach, stats and detach |
| module/ | Legacy TCP transport |
| tests/, bench/ | Correctness and controlled benchmarks |
| docs/findings/, bench/results/ | Findings and reproducible evidence |
| paper/ | Four-page report source and PDF |

Synthetic MLP, host-driven Python, handwritten Qwen and experimental variants
remain controls with their own evidence. Older bench/run_* scripts manage
hooks/pins and may terminate matching executors; read their lifecycle before
running alongside another service. The current walkthrough uses explicit
process and cleanup steps.

## Name migration

BGS_/bgs_ abbreviate bpf-gpu-server and keep BPF object names short.
The project now uses the following source and command names. Archived raw data
and historical findings keep their original commands so their records remain
replayable; they describe the same project under its former name.

| Historical name | Current name |
|---|---|
| bpfusion_load | bpf_gpu_server_load |
| fusion.bpf.c / fusion.bpf.o | server.bpf.c / server.bpf.o |
| bpfusion_queue.h / bpfusion_admission.h | bpf_gpu_server_queue.h / bpf_gpu_server_admission.h |
| bfusion_tx.ko | bpf_gpu_server_tx.ko (legacy transport) |
| BF_* macros and environment variables; bf_* symbols | BGS_*; bgs_* |
| /sys/fs/bpf/bpfusion_* pins | /sys/fs/bpf/bgs_* pins |
| bfns1/bfns2 and bfv0/bfv1 in the veth example | bgsns1/bgsns2 and bgsv0/bgsv1 |

Rebuild the loader, BPF objects, executor and any caller of the graph C API
together. Stop and detach an old instance with its own loader before switching;
old/new pin names and graph exports are not aliases. Wire framing and shared
page layout are unchanged. Existing Workspace and automation identities are
operational records and are not renamed by this source change.

## Contributing

Development happens on `main`: commit validated work directly to `main` and push;
create no other branches.

- Exercise the surface you change (run the affected client/executor/bench) before
  committing; pure documentation changes need only the links/commands checked.
- Record measurements as a [docs/findings/NNNN-*.md](docs/findings) file that
  cites the raw log path it comes from, and state the proven scope and the
  honest limitations — never claim unmet numbers.
- Update this README and [docs/architecture.md](docs/architecture.md) in the same
  commit when components, commands, or proven scope change.
- Do not commit build artifacts (`build/` is ignored) or benchmark logs other
  than the ones a finding cites.

## License

Apache-2.0 — see [LICENSE](LICENSE). The separately loaded kernel module
`module/bpf_gpu_server_tx.c` is GPL-2.0-only, as marked in its SPDX header; it uses
GPL-exported Linux TCP internals and does not link into the userspace objects.
