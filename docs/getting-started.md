# Getting started

Run the existing module-free vLLM/eBPF server and its matched native
alternative on the qualified stack. This is a research-prototype walkthrough,
not a production installer or arbitrary-kernel compatibility claim.

The [actual onboarding receipt](../bench/results/onboarding_20261016/) records
fresh Python installation, ordinary builds, and real HTTP/TCP requests for both
models and transports. This is a tested-host walkthrough, not a fresh-OS test.

## Prerequisites

Demonstrated stack:x86-64 Linux 7.3-rc3 with BTF, cgroup v2, bpffs, io_uring BPF
struct_ops/kfuncs and matching kernel UAPI headers; RTX 5090, CUDA toolkit 13.0(nvcc 13.0.88) and
driver 610.57.04; Python 3.12, system libbpf 1.3 headers and the libbpf 1.6.3 library.
See [host](../bench/results/HOST.md) and [kernel boundary](architecture.md#capacity-and-compatibility).
Use an existing root shell or privileged development container with GPU/BPF/
cgroup/bpffs access. These steps do not change the host kernel or driver.

Current pins and TCP port 39403 are shared by prototype instances:finish an
existing bpf-gpu-server run before attaching another. Keep the documented shell
environment active for loader, server and client.

Debian/Ubuntu build tools:

~~~sh
apt-get install build-essential clang llvm libbpf-dev libelf-dev zlib1g-dev \
  python3-venv python3-pip curl ca-certificates
~~~

CUDA and the running kernel's development headers must already be installed
through the machine's normal administration path. Make discovers installed
UAPI headers; missing recent io_uring definitions are a compatibility failure.

## Source and dependencies

Use your existing checkout, or obtain the public project:

~~~sh
git clone --depth 1 https://github.com/yunwei37/bpf-gpu-server.git source
cd source
~~~

Install the libbpf 1.6.3 library privately, without replacing system libraries.
Keep the system 1.3 headers on the compiler include path: adding the private
1.6.3 headers via C_INCLUDE_PATH or CPLUS_INCLUDE_PATH conflicts with
bpf_stream_vprintk in the checked-in vmlinux.h. The
[artifact compatibility note](../ARTIFACT.md#live-reruns) retains the
failed build and the passing ordinary-build smokes. A distribution with different
headers is not qualified by this walkthrough. The demonstrated x86-64 private
library install layout is usr/lib64:

~~~sh
bgs_deps="$(mktemp -d /tmp/bpf-gpu-server-deps.XXXXXX)"
curl -L https://codeload.github.com/libbpf/libbpf/tar.gz/refs/tags/v1.6.3 \
  -o "$bgs_deps/libbpf.tar.gz"
tar -xzf "$bgs_deps/libbpf.tar.gz" -C "$bgs_deps"
make -C "$bgs_deps/libbpf-1.6.3/src" \
  DESTDIR="$bgs_deps/install" install
export LIBRARY_PATH="$bgs_deps/install/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$bgs_deps/install/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

python3 -m venv --without-pip "$bgs_deps/venv"
python3 -m pip --python "$bgs_deps/venv/bin/python" install -r requirements.txt
. "$bgs_deps/venv/bin/activate"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
~~~

Pinned vLLM/Torch/Transformers/CUDA-Python versions form the tested interop
contract. The in-process setting preserves graph/tensor ownership at bootstrap.
Toolkit and Python CUDA runtime versions are distinct; Make binds against the
loaded Torch runtime. System Python packages remain unchanged.

## Build and model

~~~sh
make build/server.bpf.o build/bpf_gpu_server_load \
  vllm-mixed-uring-loop vllm-mixed-user-network-wait
model="$(python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775"))')"
~~~

Downloads use the normal Hugging Face cache; weights do not enter Git.
SM_ARCH defaults to sm_120. Other devices are not qualified here.
For the other tested checkpoint, substitute:

~~~sh
model="$(python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download("HuggingFaceTB/SmolLM2-135M", revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"))')"
~~~

Default capacity is 8. Different capacities need separate BUILD directories
and matching BPF, loader, client and executor ABI. The explicit 16-bank result
is a separate vLLM configuration, not an interchangeable module/map ABI.
After the connection-reclamation update, rebuild every shared-page consumer
with the current header; old BPF/loader/executor/module objects are incompatible.
See the [ABI and EOF ownership contract](architecture.md#framing-queue-ownership-and-publication).

## Run the kernel-owned path

~~~sh
./build/bpf_gpu_server_load stream-uring-attach
./build/vllm_mixed_uring_loop "$model" 120
~~~

Wait for the line starting with:

~~~text
resident vLLM ready:
~~~

Its remainder contains actual dimensions. Bootstrap can take minutes.
The 120 argument bounds serving time after bootstrap; Ctrl-C/SIGTERM stops the
current server. This explicit BPF-loop target needs no bpf_gpu_server_tx module.
The explicit mixed targets combine prefill and decode in the same upstream
graph batch. Older vllm_graph and homogeneous/control targets remain available
with their historical semantics; the plain vllm_graph target uses the legacy
module transport.

In another terminal, activate the same environment and set the same model path:

~~~sh
python3 tools/llm_tcp_client.py --model "$model" --http --gen 32 --rounds 1
python3 tools/llm_tcp_client.py --model "$model" --http --gen 1 --rounds 1
python3 tools/llm_tcp_client.py --model "$model" --gen 32 --rounds 1
~~~

Pass dependency/model variables to that shell, or activate its environment and
set the model path there. The client prints decoded text and arrival TTFT/
TPOT; TPOT is undefined for one output token. Short/invalid replies or invalid
lengths exit nonzero. HTTP and binary requests use the same token contract.

Input and requested output are each 1..64 tokens. Clients tokenize and
detokenize; the server does not accept text/JSON prompts. Generation has a
fixed output count and need not stop at EOS.

## Stop and clean up

After the server has stopped:

~~~sh
./build/bpf_gpu_server_load stats
./build/bpf_gpu_server_load stream-detach
rm -f /sys/fs/bpf/bgs_ctl /sys/fs/bpf/bgs_stats /sys/fs/bpf/bgs_db
~~~

Remove the disposable dependency directory only after all processes using it
have exited. This path neither loads nor unloads the legacy TCP module.

## Matched native alternative

After the kernel server and its pins are cleaned up:

~~~sh
./build/vllm_mixed_user_network_wait "$model" 120
~~~

Run the same clients. This alternative needs no BPF attach or custom module.
GPU-controlled model work is retained; network/admission uses efficient native
epoll. Demo timings are not controlled benchmark results.
[Evaluation](evaluation.md) defines matched measurement and limitations.

## Qualified 32-bank mixed profile

Stop the previous server and clean its pins before switching profiles.
For sustained mixed serving, use this separately built32-bank constructor.
Both checkpoints pass [protocol/replay and cancellation](../bench/results/resident_queue_relief_20261017/qualification.md).
The [completed five-block comparison](findings/0080-resident-queue-relief.md)
reduces completionP99 from3337ms to217ms at192requests/s; contemporaneous
optimized official vLLM is406ms. All172800requests complete without failures.
This is a shared capacity fix,not an eBPF-specific speed claim.

~~~sh
make BUILD=build/mixed32 CPPFLAGS=-DBGS_LLM_SLOTS=32 \
  build/mixed32/server.bpf.o build/mixed32/bpf_gpu_server_load \
  vllm-mixed-kernel-network vllm-mixed-user-network-wait
./build/mixed32/bpf_gpu_server_load stream-uring-attach
./build/mixed32/vllm_mixed_kernel_network "$model" 120
~~~

Use the same token client. After the kernel server exits,clean its own pins:

~~~sh
./build/mixed32/bpf_gpu_server_load stats
./build/mixed32/bpf_gpu_server_load stream-detach
rm -f /sys/fs/bpf/bgs_ctl /sys/fs/bpf/bgs_stats /sys/fs/bpf/bgs_db
~~~

Then run the matching native alternative:

~~~sh
./build/mixed32/vllm_mixed_user_network_wait "$model" 120
~~~

Use the matching loader/BPF/native/GPU objects from this one directory.
The library reports its actual capacity and bootstrap derives graph/KV sizes
from it. Do not mix8/16/32 page layouts. Global legacy defaults remain8;
the older Python token executor still assumes that layout. This profile
qualifies modern mixed serving for these two checkpoints only. It captures
more graphs and allocates more private KV banks; token gaps/TPOT need not
improve with wider batching. Maximum throughput and arbitrary-model support
are not established by the fixed192/s comparison.

## Experimental kernel-owned networking

[Finding0079](findings/0079-kernel-network-source-integration.md) provides the
0077 network-placement candidate through ordinary Make targets:

~~~sh
make build/server.bpf.o build/bpf_gpu_server_load vllm-mixed-kernel-network
./build/bpf_gpu_server_load stream-uring-attach
./build/vllm_mixed_kernel_network "$model" 120
~~~

Use the same client and stop/cleanup steps above. Default capacity is8.
The BPF loop owns io_uring CQ/SQ,connections and replies; model GPU computation
is reused and there is no GPU networking CTA. This is an explicit experimental
alternative:0078 finds no throughput advantage over the same-executor native
path, and0077 records higher CPU than the selected oldloop. Existing target
names keep their selected semantics. Use a separate BUILD plus matching
CPPFLAGS=-DBGS_LLM_SLOTS=16 only when reproducing the recorded16-slot experiment.

## Correctness and experiment reproduction

[Ordinary integration](../bench/results/batch_capacity_integration_20261015/)
records complete default 8/explicit 16 protocol and ordinary lifecycle evidence.
The [mixed scheduling archive](../bench/results/mixed_batching_20261015/) retains
the plan, review, actual sources, raw data, analysis and figures; its README gives
portable standard-library reanalysis commands. The selected ordinary mixed
adapter and default8 protocol/cancellation qualification are recorded in
[the integration record](findings/0068-mixed-source-integration.md). Historical acquisition scripts
retain their real Workspace/model-cache paths, not a portable installer.

For the separate HF/legacy-module regression:

~~~sh
make qwen-graph qwen-graph-user-network-wait
make -C module KDIR=/lib/modules/"$(uname -r)"/build CC=gcc-15
python3 tests/resident_qwen.py --executor ./build/qwen_graph --gen 64 --prefill-edges
~~~

The recorded kernel used gcc-15; match the compiler of the running kernel.
That regression manages hooks/module state and needs matching headers. Read its lifecycle first. Record failing commands/logs for
BPF, CUDA, permissions or dependency failures; there is no silent transport
fallback. Supported scope follows actual qualified runs.
