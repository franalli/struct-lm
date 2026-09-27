#!/usr/bin/env bash
# General-capability regression suite. Flags are frozen: only deltas between rows matter
# (5-shot everywhere, so numbers aren't comparable to model-card HellaSwag/GSM8K settings).
# usage: eval/run_lm_eval.sh <hf id or merged checkpoint path> <run-name> [results-dir]
#
# Why these three tasks: domain training (CPT especially) can erode general ability, and each
# task catches a different kind of loss. MMLU covers broad knowledge, GSM8K multi-step
# arithmetic reasoning (generative, so it also catches broken formatting), HellaSwag
# commonsense completion (log-likelihood, so it isolates the language model from output format).
#
# Output: <results-dir>/lm_eval/<run-name>/<model-slug>/results_<timestamp>.json.
# run_eval.py --lm-eval-dir reads the newest one and copies the three headline numbers
# into metrics.json and the run's row in results/table.md, so use the same run name for both.
set -euo pipefail
MODEL=${1:?usage: run_lm_eval.sh <model> <run-name> [results-dir]}
RUN=${2:?run name required}
OUT=${3:-results}/lm_eval/$RUN
# Mistral 3 checkpoints tokenize with Tekken via mistral-common; TOKENIZER_MODE=auto for others.
TOKENIZER_MODE=${TOKENIZER_MODE:-mistral}
# CHAT=1 for instruct/chat checkpoints: wrap prompts in the chat template, few-shot as turns.
# Changes the prompt format, so only compare CHAT=1 rows with other CHAT=1 rows.
CHAT_ARGS=()
if [[ ${CHAT:-0} == 1 ]]; then CHAT_ARGS=(--apply_chat_template --fewshot_as_multiturn); fi
mkdir -p "$OUT"

# model_args are passed straight to vllm.LLM:
#   gpu_memory_utilization=0.85  leaves headroom for lm-eval's own tensors next to vLLM's KV cache
#   max_model_len=4096           the longest 5-shot MMLU prompts fit; a smaller KV cache runs faster
#   seed=0                       with --seed 0 below, reruns of a checkpoint give identical numbers
# --batch_size auto lets vLLM schedule requests itself; it is not a fixed batch.
lm_eval --model vllm \
  --model_args "pretrained=$MODEL,tokenizer_mode=$TOKENIZER_MODE,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,seed=0" \
  --tasks mmlu,gsm8k,hellaswag \
  --num_fewshot 5 \
  --batch_size auto \
  --seed 0 \
  "${CHAT_ARGS[@]}" \
  --output_path "$OUT"

echo "results -> $OUT"
