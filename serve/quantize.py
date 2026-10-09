"""Quantize a merged checkpoint for serving (Stage 6) with llm-compressor: FP8 W8A8 (FP8_DYNAMIC: per-
channel weight scales, dynamic per-token activation scales, no calibration data) or INT4 W4A16
(GPTQ, the W4A16 preset: group 128, symmetric), calibrated on the SFT set's own chat token ids.

  python serve/quantize.py --model /vol/checkpoints/dpo-strict --scheme fp8 \
      --out /vol/checkpoints/dpo-strict-fp8
  python serve/quantize.py --model /vol/checkpoints/dpo-strict --scheme w4a16 \
      --out /vol/checkpoints/dpo-strict-w4a16 --calib data/sft/train.jsonl
  On Modal: serve/modal_serve.py::quantize. It has its own image: llmcompressor 0.14.0 (the release
  whose transformers range covers the trainer's 5.16.1) needs compressed-tensors 0.19, vLLM 0.29
  pins 0.17, so the two never share an environment (pyproject's `quantize` extra).

What it keeps from merge.py, because a quantized checkpoint must load in vLLM exactly like the bf16
one it came from:
  - the full Mistral3ForConditionalGeneration (common.auto_model_class), lm_head kept untied
    (common.keep_untied: Mistral3Config defaults tie_word_embeddings to True);
  - only the language model's Linear layers quantized: lm_head, the Pixtral vision tower and the
    projector stay bf16 (rule 10's LoRA scope; llm-compressor's own mistral3 example ignores the
    same three);
  - after the save, merge.py's post-save steps: the source's non-weight files copied (tekken.json
    for tokenizer_mode="mistral", processor_config.json, the tokenizer files; never re-saved through
    transformers), "apply_yarn_scaling": false in text_config.rope_parameters (without it vLLM 0.29
    applies YaRN attention scaling: perplexity 7.23 instead of 6.89), the tokenizer files byte-identical,
    the weight names the source's (modulo compressed-tensors' scale / packing suffixes), one weight set.
Writes <out>/quantize_meta.json: recipe, versions, calibration ids, source and output digests.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

for _d in (Path(__file__).resolve().parents[1] / "train", Path("/root/train")):
    if (_d / "merge.py").exists():  # repo checkout, or the Modal image's /root/train
        sys.path.insert(0, str(_d))
        break
import torch
from common import auto_model_class, keep_untied
from merge import check_tokenizer, copy_base_files, no_yarn_attention_scaling, weight_names

# llm-compressor matches these against module names ("re:" = regex), as loaded by transformers 5
# (model.language_model.layers..., model.vision_tower..., lm_head)
IGNORE = ["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"]
# compressed-tensors stores a quantized Linear's weight under these names next to (or instead of)
# ".weight": FP8 keeps "weight" (fp8) + "weight_scale"; W4A16 writes weight_packed / weight_scale /
# weight_shape (+ weight_zero_point if asymmetric)
CT_SUFFIX = re.compile(
    r"\.(weight_packed|weight_scale|weight_shape|weight_zero_point|weight_g_idx|input_scale)$"
)


def h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def decoder_layer_class(model) -> str:
    """The text decoder layer's class name, which GPTQ's sequential pipeline calibrates one at a
    time (oneshot's sequential_targets). Read from the model, not assumed: llm-compressor's mistral3
    example names MistralDecoderLayer, for Mistral Small 3.1."""
    for name, mod in model.named_modules():
        if re.search(r"language_model\.(model\.)?layers\.0$", name):
            return type(mod).__name__
    raise SystemExit("no language_model.layers.0 module: not a Mistral 3 layout")


def calibration(path: Path, n: int, max_len: int, model_dir: str) -> tuple[list[dict], list[str]]:
    """n SFT train records (the replay format excluded: general chat, not the deployment's), picked
    by a fixed hash of their ids, as the trainer encodes them (train/sft_data.encode: mistral-common,
    the eval's --chat rendering, prompt and answer), cut at max_len tokens."""
    import sft_data

    recs = [r for r in sft_data.load(path) if r["format"] != "replay"]
    if len(recs) < n:
        raise SystemExit(f"{path}: {len(recs)} non-replay records, {n} asked for")
    recs = sorted(recs, key=lambda r: h(f"calib:{r['eid']}"))[:n]
    rows = []
    for r in recs:
        ids = sft_data.encode(r, model_dir)["input_ids"][:max_len]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return rows, [r["eid"] for r in recs]


def check_quantized(src: Path, out: Path, scheme: str) -> dict:
    """The saved checkpoint is the source's, quantized where asked and nowhere else."""
    cfg = json.loads((out / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    if text.get("rope_parameters", {}).get("apply_yarn_scaling") is not False:
        raise SystemExit(f"{out}: text_config.rope_parameters.apply_yarn_scaling isn't false")
    for c in (cfg, text):
        if c.get("tie_word_embeddings") is not False:
            raise SystemExit(f"{out}: tie_word_embeddings isn't false (lm_head would be dropped)")
    qc = cfg.get("quantization_config")
    if not qc:
        raise SystemExit(f"{out}: no quantization_config")
    weights = [g["weights"] for g in qc["config_groups"].values()]
    if scheme == "w4a16":
        for w in weights:
            if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 128, True):
                raise SystemExit(f"{out}: not W4A16 group 128 symmetric: {w}")
    else:
        for w in weights:
            if (w["num_bits"], w["type"]) != (8, "float"):
                raise SystemExit(f"{out}: not FP8 weights: {w}")

    # one weight set: vLLM with config_format=hf loads every *.safetensors in the dir
    index = out / "model.safetensors.index.json"
    files = {p.name for p in out.glob("*.safetensors")}
    listed = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else files
    if files != listed or any(out.glob("*.bin")) or any(out.glob("*.pt")):
        raise SystemExit(f"{out}: weight files {sorted(files)} != the index's {sorted(listed)}")

    got, want = weight_names(out), weight_names(src)
    mapped = {CT_SUFFIX.sub(".weight", n) for n in got}
    if mapped != want:
        raise SystemExit(
            f"{out}: weight names differ from {src}: missing {sorted(want - mapped)[:3]}, "
            f"extra {sorted(mapped - want)[:3]}"
        )
    quantized = sorted({CT_SUFFIX.sub("", n) for n in got if CT_SUFFIX.search(n)})
    stray = [n for n in quantized if re.search(r"lm_head|vision_tower|multi_modal_projector", n)]
    if stray:
        raise SystemExit(f"{out}: ignored modules were quantized: {stray[:3]}")
    if not quantized or not all("language_model" in n for n in quantized):
        raise SystemExit(f"{out}: quantized modules outside the language model: {quantized[:3]}")
    return {"quantized_modules": len(quantized), "tensors": len(got)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged bf16 checkpoint dir")
    ap.add_argument("--scheme", required=True, choices=("fp8", "w4a16"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="data/sft/train.jsonl", help="w4a16 only")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()
    import compressed_tensors
    import llmcompressor
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from merge_check import digest

    src, out = Path(args.model), Path(args.out)
    if out.resolve() == src.resolve():
        ap.error("--out must differ from --model")
    if out.exists():  # rebuilt from scratch, as merge.py: no files from an earlier attempt
        shutil.rmtree(out)

    model = auto_model_class(str(src)).from_pretrained(src, dtype=torch.bfloat16)
    keep_untied(model)
    lm_head, embed = model.get_output_embeddings().weight, model.get_input_embeddings().weight
    if lm_head is embed or torch.equal(lm_head, embed):
        raise SystemExit(f"{src}: lm_head is tied to the input embeddings")

    meta = {"scheme": args.scheme, "source": str(src), "ignore": IGNORE}
    if args.scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
        oneshot(model=model, recipe=recipe)
    else:
        from datasets import Dataset
        from transformers import AutoTokenizer

        rows, eids = calibration(Path(args.calib), args.n_calib, args.max_len, str(src))
        layer = decoder_layer_class(model)
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)
        oneshot(
            model=model,
            # oneshot builds a processor when a dataset is given; the data is already tokenised,
            # so the source's tokenizer only has to load (it is never saved)
            processor=AutoTokenizer.from_pretrained(src),
            dataset=Dataset.from_list(rows),
            recipe=recipe,
            max_seq_length=args.max_len,
            num_calibration_samples=len(rows),
            shuffle_calibration_samples=False,
            sequential_targets=[layer],
        )
        lens = [len(r["input_ids"]) for r in rows]
        meta["calibration"] = {
            "file": args.calib,
            "n": len(rows),
            "max_len": args.max_len,
            "tokens": sum(lens),
            "max_tokens": max(lens),
            "eids_sha256": h("\n".join(eids)),
            "pick": "first n non-replay records by sha256('calib:' + eid)",
        }
        meta["sequential_target"] = layer
    meta["recipe"] = str(recipe)

    model.save_pretrained(out, save_compressed=True)
    no_yarn_attention_scaling(out)
    copied = copy_base_files(str(src), out)
    check_tokenizer(str(src), out)
    meta["check"] = check_quantized(src, out, args.scheme)
    meta["copied_from_source"] = copied
    meta["versions"] = {
        "llmcompressor": llmcompressor.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    meta["source_digest"] = digest(src)["sha256"]
    meta["digest"] = digest(out)  # before quantize_meta.json exists, so it isn't in the listing
    (out / "quantize_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9
    print(f"{args.scheme}: {src} -> {out} ({size:.2f} GB weights, {meta['check']})")


if __name__ == "__main__":
    main()
