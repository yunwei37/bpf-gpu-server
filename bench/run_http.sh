#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server: external HTTP baseline (vLLM OpenAI-compatible server).
#
#   bench/run_http.sh [model] [gen] [rounds] [port]
#
# The fair "ordinary serving stack" comparison: a standard OpenAI-compatible
# HTTP server (vLLM) on the same model, measured by the same TTFT/TPOT
# definition as the bpf-gpu-server clients. Requires vLLM installed in the current
# Python environment (`python3 -m vllm.entrypoints.openai.api_server`).
set -u
cd "$(dirname "$0")/.."

MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
GEN=${2:-32}
ROUNDS=${3:-8}
PORT=${4:-8000}
OUT=bench/results
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUT/http_${STAMP}.txt"

pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
sleep 1
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --dtype float16 --max-model-len 512 \
  --port "$PORT" > /tmp/bgs_http_srv.log 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "localhost:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 1
done

{
  echo "== bpf-gpu-server HTTP baseline (vLLM) $STAMP =="
  echo "host: $(uname -sr)  $(nproc) cpus"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "model: $MODEL (fp16, max_model_len 512, vLLM default scheduler)"
  echo "params: gen=$GEN rounds=$ROUNDS port=$PORT"
  echo
  python3 tools/llm_http_client.py --model "$MODEL" --gen "$GEN" --rounds "$ROUNDS" \
    --port "$PORT" 2>&1
} | tee "$LOG"

kill "$SRV" 2>/dev/null
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
echo "wrote $LOG"
