#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_e4_prefix_cache.sh off
#   ./run_e4_prefix_cache.sh on
#
# IMPORTANT:
# Start the vLLM server with the matching cache configuration first:
#
#   ./start_vllm.sh off
#   ./start_vllm.sh on

MODE="${1:-}"

if [[ "$MODE" != "off" && "$MODE" != "on" ]]; then
  echo "Usage: $0 {off|on}"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_DIR="${RESULT_DIR:-$STUDY_DIR/results/L40S_2026-09-26}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "$RESULT_DIR"

if [[ "$MODE" == "on" ]]; then
  echo
  echo "===== E4: WARMING SHARED PREFIX ====="

  vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 \
    --port 8000 \
    --model "$MODEL" \
    --endpoint /v1/completions \
    --dataset-name random \
    --seed 42 \
    --num-prompts 10 \
    --random-prefix-len 1536 \
    --random-input-len 512 \
    --random-output-len 8 \
    --random-range-ratio 0 \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --temperature 0
fi

if [[ "$MODE" == "off" ]]; then
  RESULT_FILENAME="e4_prefix_cache_OFF.json"
else
  RESULT_FILENAME="e4_prefix_cache_ON.json"
fi

echo
echo "===== E4: PREFIX CACHE ${MODE^^} ====="

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --seed 42 \
  --num-prompts 100 \
  --random-prefix-len 1536 \
  --random-input-len 512 \
  --random-output-len 64 \
  --random-range-ratio 0 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "$RESULT_FILENAME"