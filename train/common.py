"""Shared config loading and model setup for every training stage.

Each stage's YAML has four sections:
  model:     init_from (HF id or a previous stage's merged checkpoint), dtype, attn impl
  lora:      peft.LoraConfig kwargs (omit the section for full fine-tuning)
  data:      stage-specific paths
  training:  kwargs for the stage's TRL/transformers *Config dataclass
"""

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def parse_config() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="dotted overrides, e.g. training.learning_rate=1e-5",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.override:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_model_and_tokenizer(model_cfg: dict):
    name = model_cfg["init_from"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = auto_model_class(name).from_pretrained(
        name,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    return model, tok


def auto_model_class(name: str):
    """Ministral 3 is Mistral3ForConditionalGeneration (text + Pixtral vision encoder), which
    transformers maps to AutoModelForImageTextToText, not AutoModelForCausalLM. We keep the
    full architecture (what vLLM serves) and train on text only; LoRA targets in the configs
    are anchored on `language_model` so the vision tower stays frozen."""
    model_type = AutoConfig.from_pretrained(name).model_type
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return AutoModelForCausalLM
    return AutoModelForImageTextToText


def lora_config(cfg: dict) -> LoraConfig | None:
    return LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]) if cfg.get("lora") else None
