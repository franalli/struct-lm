"""Turn a training output into a checkpoint vLLM loads exactly like the base model.

Needed between stages (the next stage's `init_from` points at the merged dir), before the eval
harness (it takes the merged dir as --model) and before AWQ quantization (dense weights).

  python train/merge.py --adapter checkpoints/_train/cpt-8b --out checkpoints/cpt-8b      # LoRA
  python train/merge.py --full checkpoints/_train/cpt-8b-full --out checkpoints/cpt-8b-full \
      --base mistralai/Ministral-3-8B-Base-2512                                            # full FT

Merged in fp32 and cast to bf16 once (notes/decisions.md, numeric precision): LoRA loads the base in
fp32 so W + BA is added exactly (the adapters are fp32) and rounded a single time; merging into a
bf16 base would round the delta to bf16 and then the sum again. ~34 GB of RAM for the 8B. A full
fine-tuning output is already fp32 (the master weights) and gets the same single cast.
Then every non-weight file of the base snapshot that the save didn't write is copied over:
tekken.json for tokenizer_mode="mistral", processor_config.json, which Ministral 3's processor is
built from even with images disabled, and the tokenizer files. Those come from the base, never from
the training run: training doesn't change the tokenizer, and a tokenizer re-saved through
transformers writes a stub tokenizer_config.json (TokenizersBackend, no special tokens) that vLLM's
processor can't load (lm-eval failed on it; notes/contributions.md). Never copied: the
Mistral-native params.json / consolidated.safetensors (a params.json without its weights could
steer vLLM into the native format). Finally the saved weight names must equal the base's: transformers 5 renames
Mistral 3 keys on load and reverses that on save, and this proves it did.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from common import auto_model_class
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftConfig, PeftModel
from safetensors import safe_open

NATIVE = {"params.json", "consolidated.safetensors"}  # Mistral-native format: never copied
SKIP = {"README.md", ".gitattributes", "model.safetensors.index.json"}
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="LoRA adapter dir (records its base model)")
    src.add_argument("--full", help="full fine-tuning output dir (needs --base)")
    ap.add_argument("--base", help="the base model of a --full run: hub id or merged dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    trained_dir = Path(args.adapter or args.full)
    if out.resolve() == trained_dir.resolve():
        ap.error("--out must differ from the training output it merges")
    if out.exists():  # rebuilt from scratch: a rerun must not keep files from an earlier merge
        shutil.rmtree(out)

    if args.adapter:
        base_name = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        # same class selection as training (Ministral 3 -> AutoModelForImageTextToText)
        base = auto_model_class(base_name).from_pretrained(base_name, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
        trained = args.adapter
    else:
        if not args.base:
            ap.error("--full needs --base")
        base_name, trained = args.base, args.full
        model = auto_model_class(trained).from_pretrained(trained, dtype=torch.float32)
    model.to(torch.bfloat16)
    # .to() doesn't touch the sub-configs, which would still say float32; vLLM's dtype="auto"
    # (serve_vllm.sh, the latency run) could then serve the merged model in fp32
    for cfg in (
        model.config,
        model.config.get_text_config(),
        getattr(model.config, "vision_config", None),
    ):
        if cfg is not None:
            cfg.dtype = torch.bfloat16
    model.save_pretrained(out, safe_serialization=True)
    no_yarn_attention_scaling(out)
    copied = copy_base_files(base_name, out)
    check_tokenizer(base_name, out)
    check_weights(base_name, out)
    print(f"merged {trained} -> {out} (copied from base: {', '.join(copied) or 'nothing'})")


def no_yarn_attention_scaling(out: Path) -> None:
    """Add "apply_yarn_scaling": false to a YaRN text config. vLLM 0.29 builds Ministral 3's YaRN
    from the HF config without its mscale / mscale_all_dim, so it applies YaRN's default attention
    scaling (0.1 ln(16) + 1 = 1.277 on cos and sin), which transformers (mscale ratio 1) and
    Mistral's native params.json (apply_scale: false) don't: on the base's val slice, vLLM's
    perplexity is 7.226 without this key and 6.894 with it, transformers' 6.894
    (eval/vllm_ppl.py; notes/decisions.md). transformers ignores the key."""
    path = out / "config.json"
    cfg = json.loads(path.read_text())
    rope = cfg.get("text_config", cfg).get("rope_parameters") or {}
    if rope.get("rope_type") == "yarn" and "apply_yarn_scaling" not in rope:
        rope["apply_yarn_scaling"] = False
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def base_dir(base: str) -> Path:
    """The base snapshot without its weights: a local merged dir as is, or the hub files."""
    if Path(base).is_dir():
        return Path(base)
    return Path(snapshot_download(base, ignore_patterns=[f"*{w}" for w in WEIGHTS]))


def copy_base_files(base: str, out: Path) -> list[str]:
    copied = []
    for f in sorted(base_dir(base).iterdir()):
        if f.is_file() and not (out / f.name).exists():
            if f.name in NATIVE | SKIP or f.name.endswith(WEIGHTS):
                continue
            shutil.copyfile(f, out / f.name)  # content only: HF cache files are read-only
            copied.append(f.name)
    return copied


TOKENIZER_FILES = (
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def check_tokenizer(base: str, out: Path) -> None:
    """The merged checkpoint's tokenizer files are the base's, byte for byte."""
    src = base_dir(base)
    for f in TOKENIZER_FILES:
        if (src / f).exists() and (src / f).read_bytes() != (out / f).read_bytes():
            raise SystemExit(f"{out / f} differs from the base's {f}")


def weight_names(d: Path) -> set[str]:
    index = d / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    names: set[str] = set()
    for f in d.glob("model*.safetensors"):
        with safe_open(str(f), "pt") as st:
            names |= set(st.keys())
    return names


def check_weights(base: str, out: Path) -> None:
    if Path(base).is_dir():
        want = weight_names(Path(base))
    else:
        index = hf_hub_download(base, "model.safetensors.index.json")
        want = set(json.loads(Path(index).read_text())["weight_map"])
    got = weight_names(out)
    leftovers = sorted(n for n in got if "lora_" in n or "base_layer" in n or "base_model." in n)
    if leftovers:
        raise SystemExit(f"{out}: unmerged adapter weights remain, e.g. {leftovers[:3]}")
    if got != want:
        raise SystemExit(
            f"{out}: weight names differ from {base}: {len(want - got)} missing "
            f"(e.g. {sorted(want - got)[:3]}), {len(got - want)} extra (e.g. {sorted(got - want)[:3]})"
        )
    for f in ("tekken.json", "processor_config.json"):
        if not (out / f).exists():
            print(f"WARNING: no {f} in {out}; evals need --tokenizer-mode auto")


if __name__ == "__main__":
    main()
