#!/usr/bin/env bash
# OpenAI-compatible server on :8000; serve/bench_latency.py targets it.
# Ministral 3: Tekken tokenizer via mistral-common (TOKENIZER_MODE=auto for other checkpoints),
# and no image slots, so the vision encoder doesn't reserve memory (same as run_eval.py), and the
# HF config and weights for every checkpoint, never the Mistral-native params.json path, so the base
# and merged checkpoints run the same vLLM implementation (run_eval.py explains).
#
#   bash serve/serve_vllm.sh checkpoints/awq
set -euo pipefail

MODEL=${1:?model dir or HF id}
PORT=${PORT:-8000}

exec vllm serve "${MODEL}" \
  --served-model-name struct-lm \
  --tokenizer-mode "${TOKENIZER_MODE:-mistral}" \
  --limit-mm-per-prompt '{"image": 0}' \
  --config-format hf \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --tensor-parallel-size "${TP:-1}" \
  --enable-prefix-caching \
  --seed 0
