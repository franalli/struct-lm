"""Two checks on the merge path, before a merged checkpoint is evaluated or served.

  noop   (CPU) the retrospective's no-op control: an untrained adapter (LoRA B = 0) on the start
         checkpoint, merged by merge.py exactly as a trained one is, must give the start checkpoint
         back tensor for tensor. Run before a stage's first training run, per start checkpoint.
           python train/merge_check.py noop --start checkpoints/cpt-8b-replay10 \
               --config train/configs/sft.yaml --work scratch/noop-cpt --out results/runs/noop/x.json
  check  (GPU) B5's merge control for a trained adapter: start + adapter (PEFT, unmerged) against
         the merged checkpoint, both in bf16 as served. Pass (pre-registered, Stage 3b):
           - the merged model's sft_val token-mean loss is within 0.5% of start + adapter's;
           - top-1 next-token agreement >= 99% over the completion positions of 3 val records.
         Also reported: max|merged - (start+adapter)| over those logits, the adapter's own effect
         max|(start+adapter) - start|, and their ratio, flagged above 0.05: a merge error that
         isn't well under a tenth of the adapter's effect is the thing to look at even when the
         rule passes. A 1-step adapter (the smoke run) is mostly below bf16's resolution, so its
         ratio says little; the trained runs' ratios are the ones read.
           python train/merge_check.py check --adapter checkpoints/_train/sft-from-cpt/checkpoint-154 \
               --merged checkpoints/sft-from-cpt --val data/sft/sft_val.jsonl \
               --out results/runs/sft-from-cpt/merge_check.json
Exits non-zero on a failed check, so a Modal pipeline stops before its evals.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from common import auto_model_class, lora_config, parse_config
from merge import weight_names
from safetensors import safe_open

LOSS_TOL, AGREE_MIN, RATIO_FLAG = 0.005, 0.99, 0.05
PROBES = ("closed_book", "grounded", "replay")  # the 3 val records the logits are compared on


def tensors(d: Path) -> dict[str, Path]:
    out = {}
    for f in sorted(d.glob("model*.safetensors")):
        with safe_open(str(f), "pt") as st:
            out |= {k: f for k in st.keys()}  # noqa: SIM118  (safe_open, not a dict)
    return out


def noop(args) -> dict:
    """Untrained adapter -> merge.py -> compare every tensor with the start checkpoint."""
    from peft import get_peft_model

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    cfg = parse_config(["--config", args.config])
    model = auto_model_class(args.start).from_pretrained(args.start, dtype=torch.bfloat16)
    peft = get_peft_model(model, lora_config(cfg))  # init_lora_weights: B = 0, so BA = 0
    b_norm = max(float(p.abs().max()) for n, p in peft.named_parameters() if "lora_B" in n)
    peft.save_pretrained(work / "adapter")
    del model, peft
    merge = [sys.executable, str(Path(__file__).parent / "merge.py")]
    subprocess.run(
        [*merge, "--adapter", str(work / "adapter"), "--out", str(work / "merged")], check=True
    )
    a, b = tensors(Path(args.start)), tensors(work / "merged")
    assert weight_names(Path(args.start)) == weight_names(work / "merged")
    differ, max_diff = [], 0.0
    for name, fa in a.items():
        with safe_open(str(fa), "pt") as sa, safe_open(str(b[name]), "pt") as sb:
            ta, tb = sa.get_tensor(name), sb.get_tensor(name)
        if ta.dtype != tb.dtype or not torch.equal(ta, tb):
            differ.append(name)
            max_diff = max(max_diff, float((ta.float() - tb.float()).abs().max()))
    shutil.rmtree(work)
    return {
        "check": "noop",
        "start": args.start,
        "lora_b_max_abs": b_norm,
        "tensors": len(a),
        "differ": len(differ),
        "examples": differ[:5],
        "max_abs_diff": max_diff,
        "passed": not differ and b_norm == 0.0,
    }


@torch.no_grad()
def nll(model, recs: list[dict]) -> tuple[float, int]:
    """Summed NLL and token count over the records' completion tokens."""
    total, n = 0.0, 0
    for r in recs:
        ids = torch.tensor([r["input_ids"]], device=model.device)
        mask = torch.tensor(r["completion_mask"][1:], device=model.device).bool()
        logits = model(input_ids=ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
        total -= float(lp[mask].sum())
        n += int(mask.sum())
    return total, n


@torch.no_grad()
def completion_logits(model, rec: dict) -> torch.Tensor:
    """bf16 logits at the positions that predict the completion (answer tokens and </s>)."""
    ids = torch.tensor([rec["input_ids"]], device=model.device)
    mask = torch.tensor(rec["completion_mask"][1:], device=model.device).bool()
    return model(input_ids=ids).logits[0, :-1][mask]


def check(args) -> dict:
    from peft import PeftConfig, PeftModel
    from sft_data import encode, load

    start = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    val = [{**encode(r, start), "format": r["format"]} for r in load(args.val)]
    probes = [next(r for r in val if r["format"] == f) for f in PROBES]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = auto_model_class(start).from_pretrained(start, dtype=torch.bfloat16).to(dev).eval()
    unmerged = PeftModel.from_pretrained(base, args.adapter).eval()
    merged = (
        auto_model_class(args.merged)
        .from_pretrained(args.merged, dtype=torch.bfloat16)
        .to(dev)
        .eval()
    )

    u_sum, n = nll(unmerged, val)
    m_sum, _ = nll(merged, val)
    u_loss, m_loss = u_sum / n, m_sum / n
    merge_err = effect = 0.0
    err_sum = eff_sum = 0.0
    agree = total = 0
    for r in probes:
        u = completion_logits(unmerged, r)
        m = completion_logits(merged, r)
        with unmerged.disable_adapter():
            s = completion_logits(unmerged, r)
        merge_err = max(merge_err, float((m.float() - u.float()).abs().max()))
        effect = max(effect, float((u.float() - s.float()).abs().max()))
        err_sum += float((m.float() - u.float()).abs().mean()) * u.shape[0]
        eff_sum += float((u.float() - s.float()).abs().mean()) * u.shape[0]
        agree += int((m.argmax(-1) == u.argmax(-1)).sum())
        total += u.shape[0]
    rel = abs(m_loss / u_loss - 1)
    ratio = merge_err / effect if effect else None  # None: the adapter changed nothing
    return {
        "check": "merge",
        "start": start,
        "adapter": args.adapter,
        "merged": args.merged,
        "val_tokens": n,
        "val_loss_unmerged": round(u_loss, 6),
        "val_loss_merged": round(m_loss, 6),
        "val_loss_rel_diff": round(rel, 6),
        "top1_agreement": round(agree / total, 6),
        "probe_positions": total,
        "max_abs_merge_error": round(merge_err, 4),
        "max_abs_adapter_effect": round(effect, 4),
        "ratio": None if ratio is None else round(ratio, 4),
        # the same comparison in means over every logit of the probe positions: the max is one
        # logit of ~38M, a tail statistic; the mean shows the typical size of each effect
        "mean_abs_merge_error": round(err_sum / total, 5),
        "mean_abs_adapter_effect": round(eff_sum / total, 5),
        "ratio_mean": round(err_sum / eff_sum, 4) if eff_sum else None,
        "ratio_flag": ratio is None or ratio > RATIO_FLAG,
        "rule": f"|val loss rel diff| <= {LOSS_TOL} and top-1 agreement >= {AGREE_MIN}",
        "passed": rel <= LOSS_TOL and agree / total >= AGREE_MIN,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("noop")
    a.add_argument("--start", required=True)
    a.add_argument("--config", required=True, help="the stage config whose lora section is used")
    a.add_argument("--work", required=True, help="scratch dir, deleted afterwards")
    a.add_argument("--out", required=True)
    b = sub.add_parser("check")
    b.add_argument("--adapter", required=True, help="adapter dir: the run root or checkpoint-N")
    b.add_argument("--merged", required=True)
    b.add_argument("--val", required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    res = noop(args) if args.cmd == "noop" else check(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))
    if not res["passed"]:
        raise SystemExit(f"{res['check']} check failed: {out}")
    if res.get("ratio_flag"):
        print(f"FLAG: merge error / adapter effect = {res['ratio']} (> {RATIO_FLAG}, or no effect)")


if __name__ == "__main__":
    main()
