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

YAML11_BOOLS = {"yes", "no", "on", "off", "y", "n"}


def parse_config(argv: list[str] | None = None) -> dict:
    """The YAML at --config with --override applied. argv defaults to sys.argv; the Modal app
    passes its own list to run a stage in-process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="dotted overrides, e.g. training.learning_rate=1e-5",
    )
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.override:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        # yaml.safe_load reads yes/no/on/off as booleans (YAML 1.1), which turns string options
        # like eval_strategy=no into False; here only true/false are booleans.
        node[leaf] = value if value.lower() in YAML11_BOOLS else yaml.safe_load(value)
    return cfg


def load_model_and_tokenizer(model_cfg: dict):
    name = model_cfg["init_from"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = auto_model_class(name).from_pretrained(
        name,
        dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    keep_untied(model)
    return model, tok


def keep_untied(model) -> None:
    """Mistral3Config.tie_word_embeddings defaults to True, and the 8B's config.json doesn't set
    it, so any later tie_weights() call (accelerate's FSDP2 cpu_ram_efficient_loading makes one)
    would replace the 8B's own lm_head with the input embeddings (and, in full fine-tuning,
    train one matrix for both). from_pretrained keeps them apart; this records that in the
    config. Tied weights are one Parameter object, so identity is the test: a data_ptr()
    comparison would call two meta-device tensors tied (both report 0)."""
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if out is not None and out.weight is not inp.weight:
        model.config.tie_word_embeddings = False
        model.config.get_text_config().tie_word_embeddings = False


def freeze_non_text(model) -> None:
    """Full fine-tuning trains the language model only (rule 10): the Pixtral vision tower and
    the multimodal projector stay frozen, as LoRA's language_model-anchored regex leaves them."""
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)


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
