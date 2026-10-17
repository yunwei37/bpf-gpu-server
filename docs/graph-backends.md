# Supplying a captured CUDA graph

bpf-gpu-server now has an experimental one-shot graph adapter independent of LLM
model metadata. A caller supplies one CUDA graph and device input/output buffer
per slot. The server copies opaque32-bit words from the existing shared queue,
launches the graph on the GPU, waits for its child work, publishes output and
retires requests in FIFO order. Both transports use the same graph adapter.
The computation remains in the caller's captured graph.

See [graph_backend.h](../executor/graph_backend.h) for ABI version1 and buffer
ownership. Every graph must support CUDA device launch; host callbacks, host
transfers and graph allocation/free nodes are outside this interface. Referenced
tensors/graphs must stay alive until serve returns. Capture/capability errors
are reported; there is no fallback CPU computation.

## Run the existing cuBLAS/cuFFT examples

Use the Linux/GPU/CUDA/libbpf/Python prerequisites in
[getting started](getting-started.md). Activate that environment, then:

~~~sh
make build/fusion.bpf.o build/bpfusion_load \
  graph-server-kernel-network graph-server-user-network-wait
./build/bpfusion_load stream-uring-attach
python3 examples/torch_graph_server.py build/graph_server_kernel_network.so matmul --seconds 120
~~~

Wait for the ready line with abi=1, slots=8, input_words=64, output_words=64.
For FFT, substitute fft for matmul. Stop one server and clean its pins before
starting another. After it exits:

~~~sh
./build/bpfusion_load stats
./build/bpfusion_load stream-detach
rm -f /sys/fs/bpf/bpfusion_ctl /sys/fs/bpf/bpfusion_stats /sys/fs/bpf/bpfusion_db
~~~

The matched native path needs no BPF attach:

~~~sh
python3 examples/torch_graph_server.py build/graph_server_user_network_wait.so matmul --seconds 120
~~~

The listener enables TCP_NODELAY for small streamed replies,as on the LLM
adapter; both transports use that policy. See the
[finite repair control](../bench/results/graph_network_20261017/nodelay-result.md),
including its adverse native tail and measurement limits.

The example captures upstream torch.mm (cuBLAS) or torch.fft.fft (cuFFT).
Python remains alive to own the allocations; it does not execute per-request
computation. Input/output are each64 raw float32 words: matmul takes a row-major
8x8 left matrix and multiplies by fixed W[i,j]=(8*i+j)%7-3. FFT takes32
complex64 values as interleaved real/imaginary words.

A minimal binary client, in another terminal:

~~~python
import socket, struct, time
x = [float(i % 7 - 3) for i in range(64)]
body = struct.pack("<IIIIQ", 0x514c4d51, 64, 64, 0, time.monotonic_ns())
body += struct.pack("<64f", *x)
with socket.create_connection(("127.0.0.1", 39403), timeout=15) as sock:
    sock.sendall(body)
    output = bytearray()
    while len(output) < 256:
        part = sock.recv(256-len(output))
        if not part:
            raise RuntimeError("short reply")
        output.extend(part)
print(struct.unpack("<64f", output))
~~~

HTTP uses the same binary frame with Content-Length, as in the LLM transport.
A valid frame with the wrong backend input/output count receives HTTP400 or
binary EOF. There is no tokenizer, vocabulary check, text or JSON tensor API.

## Transport ownership and older control

The primary commands above use the
[ordinary kernel-network candidate](findings/0079-kernel-network-source-integration.md):

Substitute fft for matmul in the primary example,then use the same client/cleanup.
This transport
moves io_uring CQ/SQ,connection/reply lifetime into BPF and removes the GPU I/O
CTA. It changes no caller graph,buffer,ABI or graph-dispatch instruction
relative to the matched native construction.
Default8 cuBLAS/cuFFT finite numerical/lifecycle qualification is recorded in0079.
It is an experimental control,not a selected faster transport or stable
arbitrary-kernel plugin. The autoregressive scheduler stays a separate adapter.

The older graph-server-uring-loop target remains a qualified GPU-I/O control.
It runs an independent GPU I/O CTA with a BPF submit/wait loop and the same
caller graph. It is not the primary kernel-network comparison or an io_uring
userspace baseline. See [architecture](architecture.md) for target meanings.

## Varying real library work for transport experiments

The example's optional --repeat repeats the same upstream operation within
each captured graph. Input/output stay at 64 words and the final result stays
unchanged; each operation writes the same output from the same input. The
default is 1. For example:

~~~sh
python3 examples/torch_graph_server.py build/graph_server_kernel_network.so fft --repeat 16 --seconds 120
~~~

This adds actual cuFFT work; it does not optimize or replace the library kernel.
Use the identical repetition count on native and kernel paths. The
[bounded network/CPU plan](../bench/results/graph_network_20261017/plan.md)
defines the rate/work/core sweeps,complete tails,failure accounting and
independent confirmation. The [completed result](findings/0081-small-graph-network-cpu.md)
establishes no predeclared useful CPU/P99 advantage;single-CPU GEMM P99 is
adverse. Keep the finite prototype/compatibility boundary explicit.

bench/graph_work_calibration.py uses CUDA events around host graph replays.
Its elapsed time includes host submission gaps; it is neither request latency
nor a pure device-launched-kernel timing. The served graphs contain no timing
nodes. The example's capture_graphs helper keeps the calibration and actual
caller graph construction shared; it is not a new plugin API.

For remote load generation, build the native worker in the development
environment and transfer it with its coordinator/source to a Linux x86-64
client. The client needs Python 3.12 and no GPU or Python inference package.

~~~sh
make graph-network-client
# Transfer build/graph_network_client plus bench/graph_network_client.py
# and bench/graph_network_client.cpp into the same remote directory.
python3 graph_network_client.py --host "$server_ip" --operation fft \
  --rate 1000 --seconds 6 --output /tmp/fft-remote.jsonl
~~~

The static native epoll/timerfd worker emits original binary timing/reply
records. Python coordinates the unchanged arrival plan and checks results
offline. No Python request callback or JSON encoding runs in the timed worker.
The [log format](../bench/results/graph_network_20261017/native_client_format.md)
and [client-capacity repair](../bench/results/graph_network_20261017/client_capacity.md)
retain the original failed timing prefixes and real qualification.
Run the client separately from the server for a remote test.

## Qualified boundary

[Four real lifecycle runs](findings/0067-opaque-graph-backend.md) cover both
operations and transports at default8 on the existing qualified stack. These
are finite engineering checks, not throughput, universal numerical accuracy or
arbitrary CUDA/library compatibility evidence. The on-wire names and internal
queue layout still carry historical LLM naming, and lengths are limited to64
words. There is no dynamic shape negotiation, backend registry, model loader
or stable arbitrary-kernel plugin promise. The autoregressive LLM scheduler is
a separate adapter. This small interface proves a reusable transport boundary
without rewriting cuBLAS, cuFFT or model kernels.
