#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: end-to-end benchmark.
#   bench/run_e2e.sh <seconds> <work> <batch> <cpu_gpu> <cpu_resp> <idle>
#
# Runs, against one daemon lifetime:
#   verify    - GPU result vs the CPU reference on the same weights
#   own       - one request in flight at a time (latency)
#   burst     - fixed pipeline depth (throughput)
#   paced     - a fixed offered rate (goodput / drops)
#   baseline  - plain userspace UDP echo on another port, same client modes
set -u
cd "$(dirname "$0")/.."

SECS=${1:-30}
WORK=${2:-4}
BATCH=${3:-8}
CPU_GPU=${4:-6}
CPU_RESP=${5:-7}
IDLE=${6:-0}
PORT=39400
BASE_PORT=39401
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUT/e2e_${STAMP}.txt"
N_OWN=2000
N_BURST_ROUNDS=500
BURST_DEPTH=8

pkill -f 'build/executor' 2>/dev/null
pkill -f 'build/echo_udp' 2>/dev/null
sleep 0.3
./build/bpf_gpu_server_load detach lo >/dev/null 2>&1
./build/bpf_gpu_server_load attach lo >/dev/null 2>&1 || { echo "attach failed"; exit 1; }

{
	echo "== bpf-gpu-server end-to-end run $STAMP =="
	echo "host: $(uname -sr)  $(nproc) cpus"
	echo "cmdline: $(cat /proc/cmdline)"
	echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
	echo "params: seconds=$SECS work=$WORK batch=$BATCH gpu_cpu=$CPU_GPU resp_cpu=$CPU_RESP idle=$IDLE"
	echo
	echo "--- daemon ---"
} > "$LOG"

stdbuf -oL -eL ./build/executor "$SECS" "$WORK" "$BATCH" "$CPU_GPU" \
	"$CPU_RESP" "$IDLE" >> "$LOG" 2>&1 &
DPID=$!
sleep 2

{
	echo
	echo "--- client: verify (numeric agreement with the CPU reference) ---"
	./build/client verify 0 0 "$PORT" "$WORK" 2>&1
	echo
	echo "--- client: own (one request in flight) ---"
	./build/client own "$N_OWN" 0 "$PORT" "$WORK" 2>&1
	echo
	./build/client burst "$N_BURST_ROUNDS" "$BURST_DEPTH" "$PORT" "$WORK" 2>&1
	echo
	echo "--- client: paced (2000 req/s for 3000 ms) ---"
	./build/client paced 2000 0 "$PORT" "$WORK" 3000 2>&1
} >> "$LOG" 2>&1

# Let the daemon reach its own deadline so it prints the busy/idle and
# latency summaries; killing it here would discard exactly those lines.
wait "$DPID" 2>/dev/null

{
	echo
	echo "--- page/BPF counters ---"
	./build/bpf_gpu_server_load stats 2>&1
	./build/bpf_gpu_server_load detach lo 2>&1
} >> "$LOG" 2>&1

# --- baseline: userspace echo, no GPU, no control page ------------------
{
	echo
	echo "=== baseline: userspace UDP echo (port $BASE_PORT, tc hook still attached) ==="
} >> "$LOG"
./build/echo_udp "$BASE_PORT" "$CPU_RESP" >> "$LOG" 2>&1 &
EPID=$!
sleep 1
{
	echo "--- baseline client: own (one request in flight) ---"
	./build/client own "$N_OWN" 0 "$BASE_PORT" "$WORK" 2>&1
	echo "--- baseline client: burst (pipeline depth $BURST_DEPTH) ---"
	./build/client burst "$N_BURST_ROUNDS" "$BURST_DEPTH" "$BASE_PORT" "$WORK" 2>&1
} >> "$LOG" 2>&1
kill "$EPID" 2>/dev/null
wait "$EPID" 2>/dev/null

echo "wrote $LOG"
cat "$LOG"
