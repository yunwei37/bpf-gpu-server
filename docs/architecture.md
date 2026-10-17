# bpf-gpu-server architecture

bpf-gpu-server separates Linux request/reply ownership from GPU computation.
Userspace creates sockets, maps queues and captures caller graphs at startup.
The experimental kernel-network path then advances TCP and queue lifetimes
in BPF while a resident GPU dispatcher launches existing device graphs.
A matched C++ epoll path supplies the internal performance control.

This page describes current components. Experiment-specific implementations,
retired candidates, exact historical measurements and failed attempts live in
[findings](evaluation.md#historical-evidence-index), with their source snapshots
and raw data. Archived commands describe their recorded revision, not current
default behavior.

## Service and compute boundary

~~~text
                           transport
remote TCP -> framing / admission -> shared queue -> GPU dispatcher
                 ^                                     |
                 |                                     v
             Linux reply <- completed outputs <- caller CUDA graphs
~~~

The transport owns sockets, partial reads/writes, request admission, response
ordering and reclamation. The backend owns computation and its input/output
buffers. Kernel-owned networking removes a steady userspace request worker;
Linux TCP, BPF, io-wq, IRQ/softirq and GPU control still use resources.

There are two compute adapters:

| Adapter | Contract | Qualified examples |
|---|---|---|
| Opaque one-shot graph | One caller-owned device-launchable graph and input/output buffers per slot | Unchanged cuBLAS8x8FP32 and cuFFT32complex64 |
| Autoregressive generation | Captured prefill/decode graphs, model metadata, head/sampling and private KV regions | Qwen2.5-0.5B and SmolLM2-135M through HF/upstream vLLM |

[graph_backend.h](../executor/graph_backend.h) is the experimental ABI1
declaration. Counts are1..64 opaque32-bit words; callers retain graphs and
allocations until serve returns. Host callbacks/transfers and allocation/free
nodes are outside the supported device-launch interface. The adapter reports
capture/launch errors and has no CPU computation fallback. This fixed-shape
boundary is not a stable arbitrary-CUDA plugin or an autoregressive API.
See [graph examples and commands](graph-backends.md).

The generation adapter reconstructs upstream graph metadata on the GPU and
packs prefills with active decode work. Its scheduling/KV contract remains
LLM-specific. Shared capacity or mixed-batching improvements affect both
network paths and cannot establish an eBPF-only benefit.

## Transport implementations and controls

| Explicit target family | Steady ownership | Status |
|---|---|---|
| graph-server-kernel-network / vllm-mixed-kernel-network | BPF advances io_uring CQ/SQ, connections and replies; GPU performs compute dispatch | Ordinary-source experimental candidate |
| graph-server-user-network-wait / vllm-mixed-user-network-wait | C++ timed epoll, nonblocking reads/writes; same GPU compute dispatcher | Efficient matched internal comparator |
| graph-server-uring-loop / vllm-mixed-uring-loop | Independent GPU I/O CTA publishes network work; sleepable BPF loop submits/waits | Earlier qualified module-free path |
| Older module / SQPOLL / coalesced / AMK variants | Component-specific ownership | Historical controls with separate findings |

The kernel-network and native paths use the same graph caller, per-slot
arithmetic and compiled GPU dispatch. Their queue backing, transport placement
and host work differ together; this does not isolate a syscall-only effect.
The kernel-network implementation is shared between opaque graphs and
generation through [kernel_tcp.h](../executor/kernel_tcp.h),
[kernel_tcp.bpf.c](../bpf/kernel_tcp.bpf.c) and
[kernel_tcp_state.h](../bpf/include/kernel_tcp_state.h).
It has no GPU I/O CTA. Ordinary integration and finite acceptance are in
[0079](findings/0079-kernel-network-source-integration.md).

The native path uses [userspace_tcp.h](../executor/userspace_tcp.h), including
eligible-frame FIFO admission rather than FD-ordered scanning. It also retains
GPU-controlled computation; it is not a Python per-request inference baseline.
The older GPU I/O path uses [io_uring_tcp.h](../executor/io_uring_tcp.h).
The selected mixed and graph listeners enable checked TCP_NODELAY before
listen on both transports; the graph caller's repair and adverse single-cell
native result are [retained](../bench/results/graph_network_20261017/nodelay-result.md).

Official vLLM is an external reference with its own HTTP API, scheduler,
admission, KV management and capacity. Comparing it with a resident adapter
measures the whole serving design. Comparing kernel and native resident paths
with matched computation tests the transport-placement question.

## Framing, queue ownership and publication

The canonical layout is [bpf_gpu_server_queue.h](../bpf/include/bpf_gpu_server_queue.h).
Binary frames contain a24-byte header followed by32-bit words. Content-Length
HTTP POST wraps that binary body; responses are binary words/token IDs.
This is not JSON, text tokenization, TLS or an OpenAI-compatible API.

Linux TCP supplies sequencing/reassembly. Sockmap framing validates requests
before queue admission. A finite pending FIFO decouples validated frames from
physical GPU slots. It is not infinite buffering or universally qualified
backpressure. Native admission buffers one eligible frame per peer and queues
its next frame after the preceding response retires.

Generation validates token IDs; opaque graphs validate input/output counts.
Payload is written before PENDING publication. GPU work reads only published
requests and owns slot inputs until its completion protocol permits retirement.
The one-shot dispatcher launches child graphs fire-and-forget and tail-launches
itself after child outputs complete. It copies outputs to the shared page,
publishes produced counts and retires the consecutive FIFO completion prefix.

The response path preserves ordering, including partial writes. The final
word is withheld until GPU DONE; network completion/last slot access precedes
reservation release. Local send completion is not remote TCP acknowledgement.
Disconnect, pending-frame references and GPU ownership must all be resolved
before reuse. EOF-specific peer counters permit conservative reclamation;
hash collisions can delay it. These are lifetime safeguards, not a fairness
or arbitrary-overload proof. See [0062](findings/0062-pending-admission-tails.md),
[0063](findings/0063-fair-native-admission.md) and
[0072](findings/0072-connection-reclamation.md).

GPU mapped-page reads use system-scope acquire loads, with cache-bypassing
access where required; publications use system fences and appropriate stores.
CPU/BPF publication and consumption follow the corresponding acquire/release
protocol. Volatile alone is not the ordering contract. Sources are the
authority for each field and variant; the original synthetic MLP slot table
must not be applied to the current LLM/opaque-graph ring.

## Startup, active waiting and shutdown

Userspace captures graphs, initializes buffers and network state, maps/registers
the shared page, installs the applicable BPF objects and starts resident GPU
execution. It remains alive to own allocations and handle termination.

The sleepable BPF loop submits network work and uses CQ/timer progress.
Active admission waiting avoids repeated immediate callbacks while existing
work is active; the60us timer and stop sequence remain. This reduced measured
owned-group CPU in [0074](findings/0074-active-admission-wait.md), without
establishing superiority over native or a scheduler-latency bound.

Stop first prevents new work, quiesces GPU access, drains cancellation,
timer removal, outstanding sends and closes, and only then destroys graphs,
unregisters/frees buffers and releases owned pins. Queues must each drain;
multishot CQ/SQ operation counts need not be equal. The
[runbooks](getting-started.md) identify start/stop/detach steps. Never run an
archived all-in-one benchmark beside an unrelated service sharing its pins,
port or GPU.

## Capacity and compatibility

The ordinary header defaults to8slots. A different capacity needs a separate
BUILD directory and matching BPF/loader/native/GPU objects. The32-bank mixed
constructor discovers its graph/KV count from the native library's
max_requests(), with
[normal-build commands](getting-started.md#qualified-32-bank-mixed-profile).
Legacy Python layouts still fix8; globally changing the header default would
require separate compatibility work.

[0080](findings/0080-resident-queue-relief.md) qualifies the wider constructor
and completes five blocks against16banks and official vLLM. It removes
sustained queue growth at the declared192requests/s load. Wider graph/KV memory
and slightly worse token time remain. The common capacity repair is not a
network transport gain.

The tested environment is the [qualified host stack](../bench/results/HOST.md):
Linux7.3-rc3, RTX5090, CUDA13 and matching libbpf/BTF/UAPI interfaces. A recent
kernel alone does not qualify another host. The module-free path depends on
its io_uring/BPF interfaces; it does not require the legacy TCP module.
Earlier BPF kernel-socket APIs and their UDP/context restrictions are recorded
in [0044](findings/0044-uring-bpf-loop.md), with upstream references.
[module/bpf_gpu_server_tx.c](../module/bpf_gpu_server_tx.c) remains a separate legacy control;
its presence does not describe the new kernel-network path.

## Resource accounting and limits

Removing a userspace worker is an ownership change, not proof that work
disappears. TGID CPU includes charged kernel execution; sampled TIDs/io-wq,
cgroup, host IRQ/softirq and device totals overlap. Never sum nested scopes.
Whole-host counters include foreign tenants and the observer. Service affinity
restricts only owned service threads, with no exclusive cores and no global
IRQ/softirq restriction.

The bounded [small-graph experiment](../bench/results/graph_network_20261017/plan.md)
uses the same8-slot executor, remote native epoll/timerfd clients, real library
work repetitions, load through saturation, service CPU masks and full tails.
GPU/NVML is observed only at window boundaries; missing interval device samples
are unavailable, not zero GPU activity. Its [completed confirmation](findings/0081-small-graph-network-cpu.md) meets
no predeclared useful CPU/P99 advantage criterion. Worker removal changes
ownership without establishing lower total work;the bounded performance
search is closed and the prototype/compatibility limits remain.

Broader kernels/libraries/hardware, variable shapes, long context, shared-prefix
KV caching, configurable sampling, multi-GPU, general HTTP/text/TLS and
production SLOs remain unqualified. Historical6.36ms token-gap attribution
remains unresolved. Neither unrelated model-kernel optimization nor repeating
closed experiments resolves these missing contracts automatically.
