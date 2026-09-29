#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./start_vllm.sh off
#   ./start_vllm.sh on
#
# "off" disables automatic prefix caching.
# "on" enables automatic prefix caching.
#
# Optional environment overrides:
#   HOST=127.0.0.1
#   MODEL=Qwen/Qwen2.5-7B-Instruct
#   PORT=8000
#
# Example:
#   HOST=127.0.0.1 MODEL=Qwen/Qwen2.5-7B-Instruct ./start_vllm.sh off

CACHE_MODE="${1:-off}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

case "$CACHE_MODE" in
  off)
    CACHE_FLAG="--no-enable-prefix-caching"
    ;;
  on)
    CACHE_FLAG="--enable-prefix-caching"
    ;;
  *)
    echo "Usage: $0 {off|on}"
    exit 1
    ;;
esac

echo "Starting vLLM server"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Prefix caching: $CACHE_MODE"

vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  "$CACHE_FLAG"