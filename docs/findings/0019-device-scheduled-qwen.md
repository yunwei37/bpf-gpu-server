# 0019: device-scheduled Qwen graphs and 64-token kernel TCP validation

Observed 2026-10-10 UTC in the original bpf-gpu-server Workspace, RTX 5090,
CUDA 13, Linux 7.3-rc3, PyTorch 2.14.1+cu130 and Transformers 5.19.0.
Source base is `795e6044d530a6abb0dce0c74bac0d1da29834e6` plus the graph
executor change recorded with this finding. The original exported
Qwen2.5-0.5B-Instruct fp16 weights and strict eager-HF oracle are unchanged.

## Why another executor

Finding 0016's custom WMMA reference still differs from HF at the third
prompt's output 63, despite matrix prefill and investigation of projection
rounding. The new graph variant preserves the actual HF GPU operations:
Python constructs the real model from the existing binary export and
captures computation during bootstrap. A native GPU dispatcher then reads
the same eBPF queue, schedules those graphs and publishes their outputs.
Python performs no request, decode, sampling or network work after ready.
The WMMA executor remains a separate reference; its failures and earlier
performance measurements are not relabelled as graph results.

`executor/qwen_graph.py` loads parameters directly from the binary layout,
restores the normal CPU-initialized RoPE frequencies after meta allocation,
and retains all captured graph/output/buffer owners until shutdown.
`executor/qwen_graph.cu` exports the canonical queue token bound to Python;
there is no second configured capacity. With the current 64-token bound it
captures 64 prefill shapes and 126 one-token decode positions. Shared KV
buffers cover 128 positions. A capture uses the installed DynamicCache
operations and copies each completed prefix back to those buffers on-device.
Positions and causal masks are prepared during bootstrap with the installed
HF mask function and passed through its supported mask input. No model
weights, masks, sampling rules or oracle assertions are modified.

## Device scheduling and completion

[CUDA device graph launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
requires explicit device-launch instantiation and upload. Launching kernels
must themselves execute in a graph. Permitted nodes include kernels,
device-accessible copies and memset operations; host callback nodes are not
part of this path. Tail launches execute after the current graph environment
completes, and queued tail launches are ordered.

The native bootstrap instantiates/uploads the captured model graphs and
launches one dispatcher graph on the host. Its GPU kernel waits for a
PENDING slot, validates IDs, copies them to the captured input buffer, queues
the selected model graph and then a tail replay of itself. That replay
reads the completed graph's device argmax output and publishes one token.
It schedules the next decode graph or publishes DONE and advances the queue.
Tail ordering supplies model completion before result publication and graph
reuse; a prematurely visible completion flag is not used as a reuse signal.
The host sleeps, then signals shutdown through the existing mapped stop word.

```
TCP/HTTP -> sockmap/eBPF -> shared page
  -> GPU dispatcher -> device-tail HF graph -> GPU result publication
  -> existing kernel completion poll / ordered TCP TX -> client
```

This is one host service-graph launch, with many GPU-launched model graphs
and library kernels. It is not the WMMA reference's single cooperative
model kernel. Kernel accept/drain/TX and their CPU cost remain unchanged;
this is not a GPU interrupt into Linux. Requests remain serial, and the
endpoint still consumes/emits binary token IDs rather than text/JSON.

## Reproduction and receipts

Use the existing loader/module build and binary model export from findings
0013/0015. The graph target additionally needs the installed PyTorch /
Transformers packages and Python development headers: PyTorch's default
K=1 BMM path compiles a Triton driver when warming the single-token shape.
All captures and compilation occur before ready.

```sh
make qwen-graph
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --executor ./executor/qwen_graph.py
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./executor/qwen_graph.py
```

The initial [GPU-only mask capture attempt](../../bench/results/resident_qwen_device_graph_prefill_20261010.txt)
fails on the HF mask helper's CPU literal copy during capture. Moving normal
mask preparation to bootstrap yields [exact final-prefill logits](../../bench/results/resident_qwen_device_graph_prefill_mask_20261010.txt)
over three device launches. The [GPU-only 64-token sequence probe](../../bench/results/resident_qwen_device_graph_sequence64_20261010.txt)
then matches every token of `Write a short greeting.`, including the old
output-63 difference. These probes alone do not prove the TCP/HTTP path.

The [first service startup](../../bench/results/resident_qwen_device_graph_tcp64_20261010.txt)
fails because Python.h is absent during the default K=1 BMM warmup.
Installing normal `python3-dev` in the same Workspace repairs the missing
dependency. The test now retains the original startup exception instead of
masking it with a missing-ready trace split; it claims steady-state trace
coverage only after the ready line exists.

The [full TCP/HTTP 64-token regression](../../bench/results/resident_qwen_device_graph_tcp64_headers_20261010.txt)
passes 285 valid requests / 18240 exact greedy token IDs, 259 rejections,
5000 empty connections, eight concurrent clients, persistent rejection/valid
pairs across ring wrap, six pipelined replies, half-close and invalid IDs.
The [16/17/32/64-token prompt-edge run](../../bench/results/resident_qwen_device_graph_edges64_20261010.txt)
also passes: 297 valid requests / 19008 exact tokens and 259 rejections.
Both have one host service-graph launch, respectively 18240/19008 GPU model
graph launches, graph_error=0 and no host accept/receive/send after ready in
all-thread traces. Shutdown removes the test module and map pins.

The [initial expanded edge regression](../../bench/results/resident_qwen_device_graph_full_edges64_20261010.txt)
adds minimum-length and partial-tile inputs (1/15 tokens) to 16/17/32/64.
It passes 303 valid requests / 19392 exact generated tokens and 259 rejections,
with the same churn/concurrency/pipeline/trace checks. All ten prompt cases
match at generation length 64; the minimum input exercises PyTorch's default
K=1 BMM implementation. There is one host service-graph launch, 19392 GPU
model graph launches and graph_error=0. The [build/environment receipt](../../bench/results/resident_qwen_device_graph_build_20261010.txt)
records toolchain, runtime versions and native dispatcher resources.


### Queue polling fault and final acquire repair

The earlier successes include an idle `__nanosleep(256)` hint. Removing that
unsupported tuning exposed a real [one-token timeout](../../bench/results/resident_qwen_device_graph_full_edges1_20261010.txt):
one request was PENDING with zero produced tokens and zero model launches.
[Native instruction inspection](../../bench/results/resident_qwen_device_graph_idle_sass_20261010.txt)
shows the polling backedge skipped the queue-head load and comparison.
A volatile inline `ld.global.cg` with a memory clobber also
[fails](../../bench/results/resident_qwen_device_graph_volatile_edges1_20261010.txt);
its native backedge still skips the head load. These are compiler-hoisting
receipts, not a workload timeout repaired by extending a deadline.

The final source reads queue head, publication state and shutdown through
`ld.acquire.sys.global`. [PTX memory semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
define acquire/system scope separately from cache hints. The repaired native
loop has system-scope loads and a backedge returning to the head load.
No idle delay is retained. Metadata/payload reads follow PENDING acquisition;
result stores retain system fences before produced/DONE publication.

Both final expanded-edge regressions pass on this source:
[one token](../../bench/results/resident_qwen_device_graph_acquire_edges1_20261010.txt)
serves 303 valid requests / 303 exact tokens, and
[64 tokens](../../bench/results/resident_qwen_device_graph_acquire_edges64_20261010.txt)
serves 303 valid requests / 19392 exact tokens. Each also has 259 rejections,
5000 empty connections, eight concurrent clients, persistent reuse/pipeline,
half-close and all-thread network tracing. Each has one host service-graph
launch, respectively 303/19392 device model graph launches and graph_error=0.
Module/pin cleanup passes. These final receipts establish the current source;
the preceding successes and failed alternatives remain historical evidence.

## Matched host-dispatch control

`make qwen-graph qwen-graph-control` now builds both native libraries and
executable aliases `build/qwen_graph` / `build/qwen_graph_host_launch`.
Both aliases invoke the same capture script and load the adjacent library;
`make BUILD=/path/to/output` needs no copied script or library override.
The direct `executor/qwen_graph.py` command retains its normal build library.

Both native builds contain the same dispatcher kernel with a uniform `once`
argument. Default false retains GPU idle waiting/self-tail replay. True returns
after whole-request DONE or rejection; the control host polls the same map
and launches/synchronizes one service graph per request. Model graphs, KV,
argmax, queue and kernel TX are unchanged. Decode is still GPU-tail scheduled
in both modes, so this isolates whole-request host dispatch, not host-versus-
device dispatch of every model step or an optimized external serving baseline.
The host control has a request worker; only resident meets the target property.

[Device inspection](../../bench/results/qwen_graph_dispatch_resources_20261010.txt)
finds identical dispatcher SASS/resources: 35 registers, zero stack/shared/local.
Both [resident](../../bench/results/qwen_graph_dispatch_resident_edges64_20261010.txt)
and [host control](../../bench/results/qwen_graph_dispatch_host_launch_edges64_20261010.txt)
pass all 303 valid requests / 19392 exact tokens plus 259 rejections, churn,
concurrency, persistent reuse, pipeline/half-close and all-thread network tracing.
Host service-graph launches are 1/562; device model launches are 19392 in each,
with graph_error=0 and clean teardown. The aliases are also exercised by the
[resident one-token](../../bench/results/qwen_graph_dispatch_resident_edges1_20261010.txt)
and [host one-token](../../bench/results/qwen_graph_dispatch_host_launch_edges1_20261010.txt)
regressions, each with the same lifecycle surface and 303 exact tokens.

```sh
make qwen-graph qwen-graph-control
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph
HF_HOME=/workspaces/.cache/huggingface python3 tests/resident_qwen.py --gen 64 --prefill-edges --executor ./build/qwen_graph_host_launch
```

## Remaining scope

These are correctness and lifecycle receipts for the recorded prompt cases,
not a general model-quality guarantee or measured serving speedup. No new
matched graph-dispatch performance comparison has been run. Graph capture
startup/memory, total host CPU, physical-NIC delivery and completion timing
remain to measure; finding 0018's older measurements apply to its recorded
scalar source versions. GPU text processing, continuous multi-request
batching and stronger optimized serving comparisons remain open. The full
project goal is still active.
