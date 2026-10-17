#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Saturate the pipeline and report GPU occupancy / exposed bubble vs `work`.
#   bench/run_saturation.sh <list-of-work> <batch>
set -u
cd "$(dirname "$0")/.."

WORKS=${1:-"4 16 64"}
BATCH=${2:-32}
PORT=39400
SECS=10
RATE=100000   # offered requests/s; the page must drop the excess
RUN_MS=5000
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUT/saturation_${STAMP}.txt"

{
	echo "== bpf-gpu-server saturation / bubble sweep $STAMP =="
	echo "host: $(uname -sr)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
	echo "offered=$RATE req/s for ${RUN_MS} ms at batch=$BATCH, one daemon per work value"
	echo
} > "$LOG"

for W in $WORKS; do
	pkill -x executor 2>/dev/null
	sleep 0.4
	./build/bpf_gpu_server_load detach lo >/dev/null 2>&1
	./build/bpf_gpu_server_load attach lo >/dev/null 2>&1
	{
		echo "=== work=$W ==="
		echo "--- daemon ---"
	} >> "$LOG"
	./build/executor "$SECS" "$W" "$BATCH" 6 7 0 >> "$LOG" 2>&1 &
	DPID=$!
	sleep 2
	{
		echo "--- client: paced $RATE req/s ---"
		./build/client paced "$RATE" 0 "$PORT" "$W" "$RUN_MS" 2>&1
		echo "--- client: own (latency out of the saturated queue is meaningless, \
but shows service time) ---"
		./build/client own 200 0 "$PORT" "$W" 2>&1
	} >> "$LOG" 2>&1
	wait "$DPID" 2>/dev/null
	{
		echo "--- page/BPF counters ---"
		./build/bpf_gpu_server_load stats 2>&1
		echo
	} >> "$LOG"
done

pkill -x executor 2>/dev/null
./build/bpf_gpu_server_load detach lo >> "$LOG" 2>&1
echo "wrote $LOG"
cat "$LOG"
