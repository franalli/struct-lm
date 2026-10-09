#!/usr/bin/env bash
# OpenAI-compatible server on :8000 (DEPLOY.md is the deployment reference; serve/bench_latency.py and
# serve/modal_serve.py::bench target it).
# Ministral 3: Tekken tokenizer via mistral-common (TOKENIZER_MODE=auto for other checkpoints),
# and no image slots, so the vision encoder doesn't reserve memory (same as run_eval.py), and the
# HF config and weights for every checkpoint, never the Mistral-native params.json path, so the base
# and merged checkpoints run the same vLLM implementation (run_eval.py explains). Merged and
# quantized checkpoints carry "apply_yarn_scaling": false in their config (train/merge.py,
# serve/quantize.py): without it vLLM 0.29 applies YaRN attention scaling the model doesn't use.
# Quantized checkpoints (compressed-tensors) are detected from their config: no flag.
#
#   bash serve/serve_vllm.sh /vol/checkpoints/dpo-strict
#   KV_CACHE_DTYPE=fp8 bash serve/serve_vllm.sh /vol/checkpoints/dpo-strict-fp8
#
# Optional, all off by default:
#   KV_CACHE_DTYPE  fp8: an FP8 KV cache (half the bytes per token; quality-gated separately)
#   SPEC_CONFIG     JSON for --speculative-config, e.g. n-gram lookup on grounded answers
#   QUANTIZATION    fp8: vLLM's online dynamic FP8 of a bf16 checkpoint (the fallback only)
#   MAX_NUM_SEQS    requests decoded together (64: the Stage 6 sweep's top concurrency)
set -euo pipefail

MODEL=${1:?model dir or HF id}
PORT=${PORT:-8000}
EXTRA=()
[[ -n ${KV_CACHE_DTYPE:-} ]] && EXTRA+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[[ -n ${SPEC_CONFIG:-} ]] && EXTRA+=(--speculative-config "${SPEC_CONFIG}")
[[ -n ${QUANTIZATION:-} ]] && EXTRA+=(--quantization "${QUANTIZATION}")

# 8192: the KPI eval's max_model_len; the longest bench prompt is 2,094 tokens plus a 300-token cap
exec vllm serve "${MODEL}" \
  --served-model-name struct-lm \
  --tokenizer-mode "${TOKENIZER_MODE:-mistral}" \
  --limit-mm-per-prompt '{"image": 0}' \
  --config-format hf \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --tensor-parallel-size "${TP:-1}" \
  --enable-prefix-caching \
  --seed 0 \
  "${EXTRA[@]+"${EXTRA[@]}"}"
