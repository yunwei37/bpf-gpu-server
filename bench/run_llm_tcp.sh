#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: TCP LLM path benchmark.
#   bench/run_llm_tcp.sh <seconds> <gen> <rounds>
#
# Measures, against one resident-executor lifetime:
#   direct        - same model, in-process (baseline)
#   bpf-gpu-server      - UDP -> tc ingress -> token ring -> page poll on the client
#   bpf-gpu-server-tcp  - TCP -> tc ingress -> token ring -> tokens on the accepted
#                   socket (the reply path, not the page-poll shortcut)
set -u
cd "$(dirname "$0")/.."

SECS=${1:-60}
GEN=${2:-32}
ROUNDS=${3:-8}
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUT/llmtcp_${STAMP}.txt"

pkill -f 'llm_executor.py' 2>/dev/null
sleep 0.3
./build/bpf_gpu_server_load detach lo >/dev/null 2>&1
rm -f /sys/fs/bpf/bgs_ctl /sys/fs/bpf/bgs_stats \
      /sys/fs/bpf/bgs_db 2>/dev/null
./build/bpf_gpu_server_load attach lo >/dev/null 2>&1 || { echo "attach failed"; exit 1; }

{
	echo "== bpf-gpu-server LLM TCP-path run $STAMP =="
	echo "host: $(uname -sr)  $(nproc) cpus"
	echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
	echo "model: Qwen/Qwen2.5-0.5B-Instruct (fp16, batch 1, host-Python executor)"
	echo "params: seconds=$SECS gen=$GEN rounds=$ROUNDS"
	echo
	echo "--- bench ---"
} > "$LOG"

HOME=/root PYTHONPATH=executor stdbuf -oL \
	python3 executor/llm_executor.py --seconds "$SECS" --tcp >> "$LOG" 2>&1 &
EPID=$!
for _ in $(seq 1 60); do
	grep -q 'TCP listener' "$LOG" && break
	sleep 1
done

HOME=/root PYTHONPATH=executor python3 tools/llm_bench.py \
	--tcp --gen "$GEN" --rounds "$ROUNDS" >> "$LOG" 2>&1
HOME=/root PYTHONPATH=executor python3 tools/llm_tcp_client.py \
	--gen "$GEN" --rounds "$ROUNDS" >> "$LOG" 2>&1

{
	echo
	echo "--- ingress stats ---"
	./build/bpf_gpu_server_load stats 2>&1 | head -11
} >> "$LOG"

kill "$EPID" 2>/dev/null
pkill -f 'llm_executor.py' 2>/dev/null
echo "wrote $LOG"
