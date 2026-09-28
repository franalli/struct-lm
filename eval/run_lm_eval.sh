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
# Never the chat template, for every checkpoint (base, instruct, SFT/DPO/GRPO): plain 5-shot keeps
# all rows comparable, and lm-eval's chat path renders the template to text and re-encodes it, so
# Mistral control tokens ("<s>[INST]") reach the model as ordinary text. Chat format is the KPI
# eval's job (run_eval.py --chat). See notes/decisions.md.
if [[ ${CHAT:-0} == 1 ]]; then
  echo "run_lm_eval.sh: CHAT=1 is not supported; lm-eval always runs without the chat template" >&2
  exit 2
fi
mkdir -p "$OUT"

# model_args are passed straight to vllm.LLM, as JSON (lm-eval parses a JSON object whole; its
# key=value form can't carry the nested dict below):
#   gpu_memory_utilization 0.85  leaves headroom for lm-eval's own tensors next to vLLM's KV cache
#   max_model_len 4096           the longest 5-shot MMLU prompts fit; a smaller KV cache runs faster
#   seed 0                       with --seed 0 below, reruns of a checkpoint give identical numbers
#   limit_mm_per_prompt image 0  text only (CLAUDE.md rule 3): vLLM then never builds Ministral 3's
#                                image processor, which fails on merged checkpoints (a dummy-image
#                                profiling pass: notes/contributions.md). Added 2026-09-27 for
#                                Stage 2; prompts are text either way, so scores should not move
#                                (the base-8b row is re-run with it as a check).
#   config_format hf             the HF implementation for every checkpoint (see run_eval.py): the
#                                hub base otherwise goes through vLLM's Mistral-native one, which a
#                                merged checkpoint can't, so their deltas would mix in the path.
# --batch_size auto lets vLLM schedule requests itself; it is not a fixed batch.
MODEL_ARGS=$(printf '{"pretrained": "%s", "tokenizer_mode": "%s", "dtype": "bfloat16", "gpu_memory_utilization": 0.85, "max_model_len": 4096, "seed": 0, "limit_mm_per_prompt": {"image": 0}, "config_format": "hf"}' "$MODEL" "$TOKENIZER_MODE")
lm_eval --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks mmlu,gsm8k,hellaswag \
  --num_fewshot 5 \
  --batch_size auto \
  --seed 0 \
  --output_path "$OUT"

echo "results -> $OUT"
