#!/usr/bin/env bash
# AWQ W4A16 quantization with llm-compressor. Input must be a MERGED checkpoint
# (train/merge.py), not a LoRA adapter. Output loads in vLLM without extra flags
# (compressed-tensors format is auto-detected).
#
#   bash serve/quantize.sh checkpoints/grpo-merged checkpoints/awq
set -euo pipefail

MODEL=${1:?merged model dir}
OUT=${2:?output dir}
CALIB=${CALIB:-data/sft/train.jsonl}   # calibrate on in-domain chat data, not generic text
N_CALIB=${N_CALIB:-256}

python - "$MODEL" "$OUT" "$CALIB" "$N_CALIB" <<'PY'
import sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

model_dir, out, calib, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")

ds = load_dataset("json", data_files=calib)["train"].shuffle(seed=0).select(range(n))
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

oneshot(
    model=model,
    dataset=ds,
    recipe=[AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])],
    max_seq_length=2048,
    num_calibration_samples=n,
)
model.save_pretrained(out, save_compressed=True)
tok.save_pretrained(out)
print(f"AWQ -> {out}")
PY
