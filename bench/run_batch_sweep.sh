#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Sweep the daemon's batch size and report throughput + GPU bubbles.
#   bench/run_batch_sweep.sh <work> <list-of-batch-sizes>
set -u
cd "$(dirname "$0")/.."

WORK=${1:-4}
BATCHES=${2:-"1 2 4 8 16 32 64"}
PORT=39400
SECS=12
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUT/batch_sweep_${STAMP}.txt"

{
	echo "== bpf-gpu-server batch-size sweep $STAMP =="
	echo "host: $(uname -sr)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
	echo "work=$WORK  client: 2000 rounds at each batch depth, one daemon per size"
	echo
} > "$LOG"

for B in $BATCHES; do
	pkill -x executor 2>/dev/null
	sleep 0.4
	./build/bpf_gpu_server_load detach lo >/dev/null 2>&1
	./build/bpf_gpu_server_load attach lo >/dev/null 2>&1
	./build/executor "$SECS" "$WORK" "$B" 6 7 0 >> "$LOG" 2>&1 &
	DPID=$!
	sleep 2
	{
		echo "=== batch=$B ==="
		./build/client burst 2000 "$B" "$PORT" "$WORK" 2>&1
	} >> "$LOG" 2>&1
	wait "$DPID" 2>/dev/null
	echo >> "$LOG"
done

pkill -x executor 2>/dev/null
./build/bpf_gpu_server_load stats >> "$LOG" 2>&1
./build/bpf_gpu_server_load detach lo >> "$LOG" 2>&1
echo "wrote $LOG"
cat "$LOG"
