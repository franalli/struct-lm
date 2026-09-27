"""Merge a LoRA adapter into its base weights.

Needed between stages (the next stage's `init_from` points at the merged dir) and before
AWQ quantization, which operates on dense weights.

  python train/merge.py --adapter checkpoints/sft --out checkpoints/sft-merged
"""

import argparse
import shutil
from pathlib import Path

import torch
from common import auto_model_class
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_name = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    # same class selection as training (Ministral 3 -> AutoModelForImageTextToText)
    base = auto_model_class(base_name).from_pretrained(base_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.merge_and_unload().save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    copy_mistral_files(base_name, Path(args.out))
    print(f"merged {args.adapter} -> {args.out}")


MISTRAL_FILES = ("tekken.json", "processor_config.json")


def copy_mistral_files(base: str, out: Path) -> None:
    """save_pretrained writes tokenizer.json only. vLLM's tokenizer_mode="mistral" (run_eval.py,
    run_lm_eval.sh, serve_vllm.sh) needs the base model's tekken.json, and Ministral 3's
    processor is built from processor_config.json (even with images disabled)."""
    for fname in MISTRAL_FILES:
        if (Path(base) / fname).exists():  # base is an earlier merged stage
            shutil.copy(Path(base) / fname, out / fname)
            continue
        try:
            shutil.copy(hf_hub_download(base, fname), out / fname)
        except Exception as e:  # noqa: BLE001  non-Mistral-3 base: run evals with --tokenizer-mode auto
            print(f"no {fname} for {base} ({str(e)[:80]}); use tokenizer mode 'auto'")


if __name__ == "__main__":
    main()
