# bpf-gpu-server build: eBPF objects and userspace tools.
#
#   make            # build everything into build/
#   make probes     # only the eBPF objects
#   make run-probe  # build + run the ingress-wake probe
#
# Requires: clang, libbpf, libelf, zlib, libcuda/nvcc for the executor.

BUILD    := build
override CPPFLAGS += -DBGS_BUILD_DIR='"$(abspath $(BUILD))"'
ARCH     := $(shell uname -m | sed 's/x86_64/x86/;s/aarch64/arm64/')
BPF_CFLAGS := -O2 -g -target bpf -mcpu=v3 -D__TARGET_ARCH_$(ARCH) \
              -I/usr/include/$(shell uname -m)-linux-gnu -Ibpf/include \
              -Wall -Werror -Wno-missing-declarations -Wno-unused-value
CC       ?= cc
CFLAGS   ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
LDLIBS   := -lbpf -lelf -lz -lpthread

BPF_SRCS := $(filter-out bpf/uring_loop.bpf.c bpf/kernel_tcp.bpf.c,$(wildcard bpf/*.bpf.c))
BPF_OBJS := $(patsubst bpf/%.bpf.c,$(BUILD)/%.bpf.o,$(BPF_SRCS))
TOOLS    := $(BUILD)/wake_probe $(BUILD)/client $(BUILD)/echo_udp \
            $(BUILD)/bpf_gpu_server_load $(BUILD)/perfcount

.PHONY: all probes tools executor clean run-probe run-executor

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
SM_ARCH   ?= sm_120
KERNEL_UAPI := $(firstword $(wildcard /lib/modules/$(shell uname -r)/build/include/uapi /usr/src/linux-headers-$(shell uname -r)/include/uapi))

all: probes tools executor

probes: $(BPF_OBJS)

tools: $(TOOLS)

executor: $(BUILD)/executor

$(BUILD)/%.bpf.o: bpf/%.bpf.c bpf/include/bpf_gpu_server_queue.h bpf/include/bpf_gpu_server_admission.h | $(BUILD)
	clang $(CPPFLAGS) $(BPF_CFLAGS) -c $< -o $@

$(BUILD)/wake_probe: tools/wake_probe/main.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/client: tools/client.c bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Ibpf/include $< -o $@

$(BUILD)/echo_udp: tools/echo_udp.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@
$(BUILD)/bpf_gpu_server_load: tools/bpf_gpu_server_load.c bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/perfcount: tools/perfcount.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@

$(BUILD)/executor: executor/executor.cu bpf/include/bpf_gpu_server_queue.h \
                   executor/cuda_timer.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -Iexecutor -Ibpf/include \
		$< -o $@ $(LDLIBS)


run-executor: executor
	./$(BUILD)/executor $${N:-20000} $${WORK:-4} $${BATCH:-1}

$(BUILD):
	mkdir -p $(BUILD)

run-probe: all
	sudo ./$(BUILD)/wake_probe $(BUILD)/wake_probe.bpf.o 20000 50 1
	sudo ./$(BUILD)/wake_probe $(BUILD)/wake_probe.bpf.o 20000 50 0

clean:
	rm -rf $(BUILD)

.PHONY: qwen
qwen: $(BUILD)/qwen

$(BUILD)/qwen: executor/qwen.cu bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -Ibpf/include $< -o $@ $(LDLIBS)

# Same device kernel, host dispatch once per request: a mechanism control.
.PHONY: qwen-control
qwen-control: $(BUILD)/qwen_host_launch

$(BUILD)/qwen_host_launch: executor/qwen.cu bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_HOST_LAUNCH -Ibpf/include $< -o $@ $(LDLIBS)

# Device-launched HF graphs; Python captures only during bootstrap.
.PHONY: qwen-graph
qwen-graph: $(BUILD)/qwen_graph.so $(BUILD)/qwen_graph

$(BUILD)/qwen_graph.so: executor/qwen_graph.cu bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -rdc=true -shared -Xcompiler=-fPIC \
		-Ibpf/include $< -o $@ $(LDLIBS)

# GPU-owned stock io_uring TCP; uses the same HF graph computation.
.PHONY: qwen-graph-uring
qwen-graph-uring: $(BUILD)/qwen_graph_uring.so $(BUILD)/qwen_graph_uring

$(BUILD)/qwen_graph_uring.so: executor/qwen_graph.cu executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_URING_NETWORK -rdc=true -shared -Xcompiler=-fPIC \
		-Ibpf/include $< -o $@ $(LDLIBS)

# Same concurrent GPU graph banks; host resumes once per retired request.
.PHONY: qwen-graph-control
qwen-graph-control: $(BUILD)/qwen_graph_host_launch.so $(BUILD)/qwen_graph_host_launch

$(BUILD)/qwen_graph_host_launch.so: executor/qwen_graph.cu bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_HOST_LAUNCH -rdc=true -shared -Xcompiler=-fPIC \
		-Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/qwen_graph $(BUILD)/qwen_graph_host_launch $(BUILD)/qwen_graph_user_network $(BUILD)/qwen_graph_user_network_wait $(BUILD)/qwen_graph_uring: executor/qwen_graph.py | $(BUILD)
	ln -sf $(abspath executor/qwen_graph.py) $@

# Same resident GPU graphs with an epoll userspace TCP/HTTP control.
.PHONY: qwen-graph-user-network
qwen-graph-user-network: $(BUILD)/qwen_graph_user_network.so $(BUILD)/qwen_graph_user_network

$(BUILD)/qwen_graph_user_network.so: executor/qwen_graph.cu executor/userspace_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_USER_NETWORK -rdc=true -shared -Xcompiler=-fPIC \
		-Ibpf/include $< -o $@ $(LDLIBS)

# CPU-efficient network control; model graphs and device dispatcher are unchanged.
.PHONY: qwen-graph-user-network-wait
qwen-graph-user-network-wait: $(BUILD)/qwen_graph_user_network_wait.so $(BUILD)/qwen_graph_user_network_wait

$(BUILD)/qwen_graph_user_network_wait.so: executor/qwen_graph.cu executor/userspace_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_USER_NETWORK -DBGS_USER_NETWORK_WAIT -rdc=true -shared -Xcompiler=-fPIC \
		-Ibpf/include $< -o $@ $(LDLIBS)

# Upstream vLLM graphs with bootstrap-owned KV and GPU prefill/decode.
VLLM_PYTHON ?= python3
VLLM_CUDART = $(shell $(VLLM_PYTHON) -c 'import pathlib,torch; print(next(line.rsplit(" ",1)[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines() if "/libcudart.so."+torch.version.cuda.split(".")[0] in line))')
# Default uses the qualified packed prefill and complete-prefix release policy.
# Named serial/grouped controls keep causal comparisons explicit.
VLLM_DEFAULT_NAMES := vllm_graph vllm_graph_user_network_wait vllm_graph_trace vllm_graph_user_network_wait_trace
VLLM_PADDED_NAMES := vllm_graph_padded vllm_graph_padded_user_network_wait vllm_graph_padded_trace vllm_graph_padded_user_network_wait_trace
VLLM_GROUPED_NAMES := vllm_graph_grouped vllm_graph_grouped_user_network_wait vllm_graph_grouped_trace vllm_graph_grouped_user_network_wait_trace
VLLM_CHAINED_NAMES := vllm_graph_chained vllm_graph_chained_user_network_wait vllm_graph_chained_trace vllm_graph_chained_user_network_wait_trace
VLLM_URING_NAMES := vllm_graph_uring vllm_graph_uring_trace
VLLM_URING_CONTROL_NAMES := vllm_graph_uring_full_copy vllm_graph_uring_full_copy_trace
VLLM_URING_LOOP_NAMES := vllm_graph_uring_loop vllm_graph_uring_loop_trace
VLLM_URING_COALESCED_NAMES := vllm_graph_uring_loop_coalesced vllm_graph_uring_loop_coalesced_trace
VLLM_URING_LOOP_AFTER_NAMES := vllm_graph_uring_loop_after vllm_graph_uring_loop_after_trace
VLLM_NAMES := $(VLLM_DEFAULT_NAMES) $(VLLM_PADDED_NAMES) $(VLLM_GROUPED_NAMES) $(VLLM_CHAINED_NAMES) $(VLLM_URING_NAMES) $(VLLM_URING_CONTROL_NAMES) $(VLLM_URING_LOOP_NAMES) $(VLLM_URING_COALESCED_NAMES) $(VLLM_URING_LOOP_AFTER_NAMES)

.PHONY: vllm-graph vllm-graph-user-network-wait vllm-graph-trace vllm-graph-padded vllm-graph-grouped vllm-graph-release
.PHONY: vllm-graph-uring
vllm-graph-uring: $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_URING_NAMES))) $(addprefix $(BUILD)/,$(VLLM_URING_NAMES))

.PHONY: vllm-graph-uring-loop qwen-graph-uring-loop
vllm-graph-uring-loop: $(BUILD)/uring_loop.bpf.o $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_URING_LOOP_NAMES))) $(addprefix $(BUILD)/,$(VLLM_URING_LOOP_NAMES))
qwen-graph-uring-loop: $(BUILD)/uring_loop.bpf.o $(BUILD)/qwen_graph_uring_loop.so $(BUILD)/qwen_graph_uring_loop

.PHONY: vllm-graph-uring-coalesced
vllm-graph-uring-coalesced: $(BUILD)/uring_loop.bpf.o $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_URING_COALESCED_NAMES))) $(addprefix $(BUILD)/,$(VLLM_URING_COALESCED_NAMES))

.PHONY: vllm-graph-uring-loop-after
vllm-graph-uring-loop-after: $(BUILD)/uring_loop_after.bpf.o $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_URING_LOOP_AFTER_NAMES))) $(addprefix $(BUILD)/,$(VLLM_URING_LOOP_AFTER_NAMES))

$(BUILD)/uring_loop_after.bpf.o: bpf/uring_loop.bpf.c bpf/include/bpf_gpu_server_queue.h bpf/include/bpf_gpu_server_admission.h | $(BUILD)
	clang $(BPF_CFLAGS) -DBGS_IO_CQ_AFTER_SUBMIT -c $< -o $@

$(BUILD)/qwen_graph_uring_loop.so: executor/qwen_graph.cu executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_URING_NETWORK -DBGS_URING_BPF_LOOP -rdc=true -shared -Xcompiler=-fPIC \
		-I$(KERNEL_UAPI) -Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/qwen_graph_uring_loop: executor/qwen_graph.py | $(BUILD)
	ln -sf $(abspath executor/qwen_graph.py) $@

.PHONY: vllm-graph-uring-full-copy
vllm-graph-uring-full-copy: $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_URING_CONTROL_NAMES))) $(addprefix $(BUILD)/,$(VLLM_URING_CONTROL_NAMES))

.PHONY: vllm-graph-chained
vllm-graph-chained: $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_CHAINED_NAMES))) $(addprefix $(BUILD)/,$(VLLM_CHAINED_NAMES))
vllm-graph-release: vllm-graph vllm-graph-user-network-wait vllm-graph-trace vllm-graph-padded
vllm-graph-padded: $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_PADDED_NAMES))) $(addprefix $(BUILD)/,$(VLLM_PADDED_NAMES))
vllm-graph-grouped: $(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_GROUPED_NAMES))) $(addprefix $(BUILD)/,$(VLLM_GROUPED_NAMES))
vllm-graph: $(BUILD)/vllm_graph.so $(BUILD)/vllm_graph
vllm-graph-user-network-wait: $(BUILD)/vllm_graph_user_network_wait.so $(BUILD)/vllm_graph_user_network_wait
vllm-graph-trace: $(BUILD)/vllm_graph_trace.so $(BUILD)/vllm_graph_trace $(BUILD)/vllm_graph_user_network_wait_trace.so $(BUILD)/vllm_graph_user_network_wait_trace

$(addprefix $(BUILD)/,$(VLLM_NAMES)): executor/vllm_graph.py | $(BUILD)
	ln -sf $(abspath executor/vllm_graph.py) $@

$(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_NAMES))): $(BUILD)/%.so: executor/vllm_graph.cu executor/userspace_tcp.h executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) --cudart=none -dc -Xcompiler=-fPIC -Ibpf/include \
		$(if $(findstring user_network,$*),-DBGS_USER_NETWORK -DBGS_USER_NETWORK_WAIT) $(if $(findstring trace,$*),-DBGS_TRACE) $(if $(filter $*,$(VLLM_URING_NAMES) $(VLLM_URING_CONTROL_NAMES) $(VLLM_URING_LOOP_NAMES) $(VLLM_URING_COALESCED_NAMES) $(VLLM_URING_LOOP_AFTER_NAMES)),-DBGS_URING_NETWORK) $(if $(filter $*,$(VLLM_URING_LOOP_NAMES) $(VLLM_URING_COALESCED_NAMES) $(VLLM_URING_LOOP_AFTER_NAMES)),-DBGS_URING_BPF_LOOP -I$(KERNEL_UAPI)) $(if $(filter $*,$(VLLM_URING_LOOP_AFTER_NAMES)),-DBGS_IO_CQ_AFTER_SUBMIT) $(if $(findstring full_copy,$*),-DBGS_IO_FULL_COPY) $(if $(findstring coalesced,$*),-DBGS_IO_COALESCED) $(if $(filter $*,$(VLLM_DEFAULT_NAMES) $(VLLM_PADDED_NAMES) $(VLLM_CHAINED_NAMES) $(VLLM_URING_NAMES) $(VLLM_URING_CONTROL_NAMES) $(VLLM_URING_LOOP_NAMES) $(VLLM_URING_COALESCED_NAMES) $(VLLM_URING_LOOP_AFTER_NAMES)),-DBGS_PAD_PREFILL) $(if $(filter $*,$(VLLM_DEFAULT_NAMES) $(VLLM_CHAINED_NAMES) $(VLLM_URING_NAMES) $(VLLM_URING_CONTROL_NAMES) $(VLLM_URING_LOOP_NAMES) $(VLLM_URING_COALESCED_NAMES) $(VLLM_URING_LOOP_AFTER_NAMES)),-DBGS_PREFIX_RELEASE) $(if $(filter $*,$(VLLM_CHAINED_NAMES)),-DBGS_CHAINED_DISPATCH) $< -o $@.o
	$(NVCC) -arch=$(SM_ARCH) -dlink -Xcompiler=-fPIC $@.o -o $@.dlink.o
	$(CXX) -shared -Wl,-z,defs $@.o $@.dlink.o $(VLLM_CUDART) -Wl,-rpath,$(dir $(VLLM_CUDART)) -o $@ $(LDLIBS)

# Optional upstream AMK graph bootstrap; reuse the existing model/transport dispatcher.
AMK_PYTHON ?= python3
AMK_INCLUDE = $(shell $(AMK_PYTHON) -c 'import pathlib,vm; print(pathlib.Path(vm.__file__).parent)')
.PHONY: amk-graph-uring-loop amk-graph-user-network-wait
amk-graph-uring-loop: $(BUILD)/uring_loop.bpf.o $(BUILD)/amk_graph_builder.so $(BUILD)/amk_graph_uring_loop.so $(BUILD)/amk_graph_uring_loop
amk-graph-user-network-wait: $(BUILD)/amk_graph_builder.so $(BUILD)/amk_graph_user_network_wait.so $(BUILD)/amk_graph_user_network_wait

$(BUILD)/amk_graph_builder.so: executor/amk_graph.cu | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(SM_ARCH) -rdc=true -shared -Xcompiler=-fPIC -I$(AMK_INCLUDE) $< -lcuda -o $@

$(BUILD)/amk_graph_uring_loop.so: executor/qwen_graph.cu executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_URING_NETWORK -DBGS_URING_BPF_LOOP -rdc=true -shared -Xcompiler=-fPIC -I$(KERNEL_UAPI) -Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/amk_graph_user_network_wait.so: executor/qwen_graph.cu executor/userspace_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_USER_NETWORK -DBGS_USER_NETWORK_WAIT -rdc=true -shared -Xcompiler=-fPIC -Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/amk_graph_uring_loop $(BUILD)/amk_graph_user_network_wait: executor/amk_graph.py | $(BUILD)
	ln -sf $(abspath executor/amk_graph.py) $@

.PHONY: amk-graph-uring-coalesced
amk-graph-uring-coalesced: $(BUILD)/uring_loop.bpf.o $(BUILD)/amk_graph_builder.so $(BUILD)/amk_graph_uring_loop_coalesced.so $(BUILD)/amk_graph_uring_loop_coalesced

$(BUILD)/amk_graph_uring_loop_coalesced.so: executor/qwen_graph.cu executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) -DBGS_URING_NETWORK -DBGS_URING_BPF_LOOP -DBGS_IO_COALESCED -rdc=true -shared -Xcompiler=-fPIC -I$(KERNEL_UAPI) -Ibpf/include $< -o $@ $(LDLIBS)

$(BUILD)/amk_graph_uring_loop_coalesced: executor/amk_graph.py | $(BUILD)
	ln -sf $(abspath executor/amk_graph.py) $@


# Opaque captured-graph backend; the compute graph and buffers are caller-owned.
GRAPH_SERVER_NAMES := graph_server_uring_loop graph_server_user_network_wait graph_server_kernel_network
.PHONY: graph-server-uring-loop graph-server-user-network-wait
graph-server-uring-loop: $(BUILD)/uring_loop.bpf.o $(BUILD)/graph_server_uring_loop.so
graph-server-user-network-wait: $(BUILD)/graph_server_user_network_wait.so

# Experimental0077 network placement; existing targets retain the selected path.
.PHONY: graph-server-kernel-network
graph-server-kernel-network: $(BUILD)/kernel_tcp.bpf.o $(BUILD)/graph_server_kernel_network.so

$(addprefix $(BUILD)/,$(addsuffix .so,$(GRAPH_SERVER_NAMES))): $(BUILD)/%.so: executor/graph_server.cu executor/graph_backend.h executor/userspace_tcp.h executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) --cudart=none -dc -Xcompiler=-fPIC -Ibpf/include \
		$(if $(findstring user_network,$*),-DBGS_USER_NETWORK -DBGS_USER_NETWORK_WAIT,-DBGS_URING_NETWORK -DBGS_URING_BPF_LOOP $(if $(findstring kernel_network,$*),-DBGS_KERNEL_TCP) -I$(KERNEL_UAPI)) $< -o $@.o
	$(NVCC) -arch=$(SM_ARCH) -dlink -Xcompiler=-fPIC $@.o -o $@.dlink.o
	$(CXX) -shared -Wl,-z,defs $@.o $@.dlink.o $(VLLM_CUDART) -Wl,-rpath,$(dir $(VLLM_CUDART)) -o $@ $(LDLIBS)


# Qualified packed mixed prefill/decode; legacy homogeneous controls stay explicit.
VLLM_MIXED_NAMES := vllm_mixed_uring_loop vllm_mixed_uring_loop_trace vllm_mixed_user_network_wait vllm_mixed_user_network_wait_trace vllm_mixed_kernel_network vllm_mixed_kernel_network_trace
.PHONY: vllm-mixed-uring-loop vllm-mixed-user-network-wait vllm-mixed-trace
vllm-mixed-uring-loop: $(BUILD)/uring_loop.bpf.o $(BUILD)/vllm_mixed_uring_loop.so $(BUILD)/vllm_mixed_uring_loop
vllm-mixed-user-network-wait: $(BUILD)/vllm_mixed_user_network_wait.so $(BUILD)/vllm_mixed_user_network_wait
vllm-mixed-trace: $(BUILD)/uring_loop.bpf.o $(BUILD)/vllm_mixed_uring_loop_trace.so $(BUILD)/vllm_mixed_uring_loop_trace $(BUILD)/vllm_mixed_user_network_wait_trace.so $(BUILD)/vllm_mixed_user_network_wait_trace

$(addprefix $(BUILD)/,$(VLLM_MIXED_NAMES)): executor/vllm_mixed.py executor/mixed_replay.py | $(BUILD)
	ln -sf $(abspath executor/vllm_mixed.py) $@

$(addprefix $(BUILD)/,$(addsuffix .so,$(VLLM_MIXED_NAMES))): $(BUILD)/%.so: executor/vllm_mixed.cu executor/userspace_tcp.h executor/io_uring_tcp.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(NVCC) $(CPPFLAGS) -O3 -std=c++17 -arch=$(SM_ARCH) --cudart=none -dc -Xcompiler=-fPIC -Ibpf/include \
		-DBGS_PAD_PREFILL -DBGS_PREFIX_RELEASE $(if $(findstring user_network,$*),-DBGS_USER_NETWORK -DBGS_USER_NETWORK_WAIT,-DBGS_URING_NETWORK -DBGS_URING_BPF_LOOP $(if $(findstring kernel_network,$*),-DBGS_KERNEL_TCP) -I$(KERNEL_UAPI)) $(if $(findstring trace,$*),-DBGS_TRACE) $< -o $@.o
	$(NVCC) -arch=$(SM_ARCH) -dlink -Xcompiler=-fPIC $@.o -o $@.dlink.o
	$(CXX) -shared -Wl,-z,defs $@.o $@.dlink.o $(VLLM_CUDART) -Wl,-rpath,$(dir $(VLLM_CUDART)) -o $@ $(LDLIBS)


# Shared CPU-only transport state is independent of model or graph arithmetic.
.PHONY: vllm-mixed-kernel-network vllm-mixed-kernel-network-trace
vllm-mixed-kernel-network: $(BUILD)/kernel_tcp.bpf.o $(BUILD)/vllm_mixed_kernel_network.so $(BUILD)/vllm_mixed_kernel_network
vllm-mixed-kernel-network-trace: $(BUILD)/kernel_tcp.bpf.o $(BUILD)/vllm_mixed_kernel_network_trace.so $(BUILD)/vllm_mixed_kernel_network_trace

$(BUILD)/vllm_mixed_kernel_network.so $(BUILD)/vllm_mixed_kernel_network_trace.so $(BUILD)/graph_server_kernel_network.so: executor/kernel_tcp.h bpf/include/kernel_tcp_state.h

$(BUILD)/kernel_tcp.bpf.o: bpf/kernel_tcp.bpf.c bpf/include/kernel_tcp_state.h bpf/include/vmlinux.h bpf/include/bpf_gpu_server_admission.h bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	clang $(CPPFLAGS) $(BPF_CFLAGS) -c $< -o $@

.PHONY: graph-network-client
graph-network-client: $(BUILD)/graph_network_client
$(BUILD)/graph_network_client: bench/graph_network_client.cpp bpf/include/bpf_gpu_server_queue.h | $(BUILD)
	$(CXX) $(CPPFLAGS) -O3 -std=c++17 -static -Ibpf/include $< -o $@
