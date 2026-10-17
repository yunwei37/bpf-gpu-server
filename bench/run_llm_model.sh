#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: matched LLM comparison for an arbitrary model.
#   bench/run_llm_model.sh <model> <seconds> <gen> <rounds>
# Same host/process/executor for direct vs UDP vs TCP; only --model changes.
set -u
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
SECS=${2:-90}
GEN=${3:-32}
ROUNDS=${4:-6}
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG=$(echo "$MODEL" | tr '/.' '__')
LOG="$OUT/llm_${TAG}_${STAMP}.txt"

pkill -f 'llm_executor.py' 2>/dev/null
sleep 0.3
./build/bpf_gpu_server_load detach lo >/dev/null 2>&1
rm -f /sys/fs/bpf/bgs_ctl /sys/fs/bpf/bgs_stats /sys/fs/bpf/bgs_db 2>/dev/null
./build/bpf_gpu_server_load attach lo >/dev/null 2>&1 || { echo "attach failed"; exit 1; }

{
	echo "== bpf-gpu-server LLM matched run $STAMP =="
	echo "host: $(uname -sr)  $(nproc) cpus"
	echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
	echo "model: $MODEL (fp16, batch 1, host-Python executor)"
	echo "params: seconds=$SECS gen=$GEN rounds=$ROUNDS"
	echo
	echo "--- load ---"
} > "$LOG"

HOME=/root PYTHONPATH=executor stdbuf -oL \
	python3 executor/llm_executor.py --seconds "$SECS" --tcp --model "$MODEL" \
	>> "$LOG" 2>&1 &
EPID=$!
for _ in $(seq 1 120); do
	grep -q 'TCP listener' "$LOG" && break
	sleep 1
done

HOME=/root PYTHONPATH=executor python3 tools/llm_bench.py --tcp --model "$MODEL" \
	--gen "$GEN" --rounds "$ROUNDS" 2>&1 | grep -vE 'Loading weights|it/s|^$' >> "$LOG"

{
	echo
	echo "--- ingress stats ---"
	./build/bpf_gpu_server_load stats 2>&1 | head -11
} >> "$LOG"

kill "$EPID" 2>/dev/null
pkill -f 'llm_executor.py' 2>/dev/null
echo "wrote $LOG"
