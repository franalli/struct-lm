"""Two checks on the merge path, before a merged checkpoint is evaluated or served.

  noop   (CPU) the retrospective's no-op control: an untrained adapter (LoRA B = 0) on the start
         checkpoint, merged by merge.py exactly as a trained one is, must give the start checkpoint
         back tensor for tensor. Run before a stage's first training run, per start checkpoint.
           python train/merge_check.py noop --start checkpoints/cpt-8b-replay10 \
               --config train/configs/sft.yaml --work scratch/noop-cpt --out results/runs/noop/x.json
  check  (GPU) B5's merge gate for a trained adapter (amended 2026-10-06, before any downstream
         eval; notes/decisions.md). Over every --val completion position (sft_val: 11,351; for a
         DPO run, data/dpo/val.jsonl's pairs, chosen and rejected each a sequence of their
         sampled ids), three models loaded one at a time are compared at each position (argmax,
         top-1/top-2 margin, the target's log-prob):
           ref       the start in fp32 + the fp32 adapter, unmerged (the exact function)
           unmerged  the start in bf16 + the fp32 adapter: the noise floor of bf16 inference
           merged    the merged bf16 checkpoint under test
         Pass: the merge adds at most 0.1% of positions in argmax flips against ref over what
         unmerged already has (11 of 11,351 on sft_val); its mean |delta log-prob| against ref is at most 1.5x
         unmerged's; and its val loss (token-mean NLL) is within 0.5% of unmerged's. Reported, not
         gated: merged-vs-unmerged agreement (two bf16 approximations: it measures bf16's own
         near-tie noise as much as the merge), the 3-probe logit ratios, and the sha256 of every
         file of the merged checkpoint, so the gate and the evals are provably on the same file.
           python train/merge_check.py check --adapter checkpoints/_train/sft-from-cpt/checkpoint-77 \
               --merged checkpoints/sft-from-cpt --val data/sft/sft_val.jsonl \
               --out results/runs/sft-from-cpt/merge_check.json
         (The first gate, top-1 >= 99% on 3 probes / 295 positions, failed sft-from-cpt on 4 flips
         of sampling variation; `diagnose` is the measurement that showed it, now this check.)
  regate (no GPU) the amended gate applied to a run already measured by `diagnose`, with its
         checkpoint digest (`digest`): writes the amended merge_check.json, keeping the first
         gate's result as merge_check_original.json.
  digest (CPU) sha256 of each file of a merged checkpoint, and of the list.
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

LOSS_TOL, ADDED_FLIPS_MAX, LP_RATIO_MAX, RATIO_FLAG = 0.005, 0.001, 1.5, 0.05
PROBES = ("closed_book", "grounded", "replay")  # the 3 val records the logits are compared on
N_PROBES = 3


def val_records(path: str, start: str) -> list[dict]:
    """{input_ids, completion_mask, format} per val sequence. SFT records are encoded through
    mistral-common (sft_data.encode); DPO pair records (with chosen_ids) give two sequences each,
    prompt + chosen and prompt + rejected, from their sampled ids as the trainer sees them."""
    from sft_data import encode, load

    recs = load(path)
    if not recs or "chosen_ids" not in recs[0]:
        return [{**encode(r, start), "format": r["format"]} for r in recs]
    out = []
    for r in recs:
        for side in ("chosen_ids", "rejected_ids"):
            out.append(
                {
                    "input_ids": r["prompt_ids"] + r[side],
                    "completion_mask": [0] * len(r["prompt_ids"]) + [1] * len(r[side]),
                    "format": r["format"],
                }
            )
    return out


def probe_formats(val: list[dict]) -> list[str]:
    """PROBES' formats that the val set has (all three on sft_val), then its other formats in
    name order, up to 3 (DPO val has no replay)."""
    present = {r["format"] for r in val}
    first = [f for f in PROBES if f in present]
    return (first + sorted(present - set(first)))[:N_PROBES]


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
def completion_logits(model, rec: dict) -> torch.Tensor:
    """bf16 logits at the positions that predict the completion (answer tokens and </s>)."""
    ids = torch.tensor([rec["input_ids"]], device=model.device)
    mask = torch.tensor(rec["completion_mask"][1:], device=model.device).bool()
    return model(input_ids=ids).logits[0, :-1][mask]


@torch.no_grad()
def position_stats(model, recs: list[dict]) -> dict[str, torch.Tensor]:
    """At every completion position: the argmax, the top-1 minus top-2 logit margin, and the
    log-probability of the actual next token. On CPU, so models can be loaded one at a time."""
    out: dict[str, list] = {"argmax": [], "margin": [], "lp": []}
    for r in recs:
        ids = torch.tensor([r["input_ids"]], device=model.device)
        mask = torch.tensor(r["completion_mask"][1:], device=model.device).bool()
        logits = model(input_ids=ids).logits[0, :-1][mask].float()
        top = logits.topk(2, -1)
        out["argmax"].append(top.indices[:, 0].cpu())
        out["margin"].append((top.values[:, 0] - top.values[:, 1]).cpu())
        target = ids[0, 1:][mask][:, None]
        out["lp"].append(torch.log_softmax(logits, -1).gather(1, target)[:, 0].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def digest(path: str | Path) -> dict:
    """sha256 of every weight and config file of a checkpoint, and of their sorted list."""
    import hashlib

    files = {}
    for f in sorted(Path(path).glob("*")):
        if f.suffix in (".safetensors", ".json") and f.is_file():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            files[f.name] = h.hexdigest()
    listing = "".join(f"{h}  {n}\n" for n, h in files.items())
    return {
        "checkpoint": str(path),
        "files": files,
        "sha256": hashlib.sha256(listing.encode()).hexdigest(),
    }


def gate(diag: dict) -> dict:
    """The amended B5 gate on a measurement: flips added over bf16's own, log-prob error ratio,
    val loss."""
    u, m = diag["unmerged_bf16_vs_fp32_ref"], diag["merged_bf16_vs_fp32_ref"]
    added = m["flips"] - u["flips"]
    allowed = int(ADDED_FLIPS_MAX * diag["positions"])
    lp_ratio = m["mean_abs_lp_diff"] / u["mean_abs_lp_diff"]
    rel = abs(diag["nll_merged_bf16"] / diag["nll_unmerged_bf16"] - 1)
    return {
        "added_flips": added,
        "added_flips_allowed": allowed,
        "lp_error_ratio": round(lp_ratio, 4),
        "val_loss_unmerged": diag["nll_unmerged_bf16"],
        "val_loss_merged": diag["nll_merged_bf16"],
        "val_loss_rel_diff": round(rel, 6),
        # the same yardstick at every stage, in nats next to the relative line (2026-10-09)
        "val_loss_abs_diff_nats": round(
            abs(diag["nll_merged_bf16"] - diag["nll_unmerged_bf16"]), 6
        ),
        "merged_vs_unmerged_top1": diag["merged_vs_unmerged_bf16"]["top1_agreement"],
        "rule": f"added flips <= {allowed} of {diag['positions']}, merged |dlp| <= "
        f"{LP_RATIO_MAX}x unmerged's (both against fp32), sft_val loss within {LOSS_TOL:.1%} "
        f"(|merged - unmerged| reported in nats)",
        "passed": added <= allowed and lp_ratio <= LP_RATIO_MAX and rel <= LOSS_TOL,
    }


def measure(args) -> tuple[dict, dict]:
    """Position stats of ref / unmerged / merged over all of --val, and the 3 probes' logits
    (unmerged, the start without the adapter, merged), one model on the GPU at a time."""
    import gc

    from peft import PeftConfig, PeftModel

    start = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    val = val_records(args.val, start)
    probes = [next(r for r in val if r["format"] == f) for f in probe_formats(val)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logits: dict[str, list] = {}

    def run(name: str, dtype, adapter: bool, path: str) -> dict:
        m = auto_model_class(path).from_pretrained(path, dtype=dtype).to(dev).eval()
        if adapter:
            m = PeftModel.from_pretrained(m, args.adapter).eval()
        st = position_stats(m, val)
        if dtype == torch.bfloat16:
            logits[name] = [completion_logits(m, r).float().cpu() for r in probes]
            if adapter:
                with m.disable_adapter():
                    logits["start"] = [completion_logits(m, r).float().cpu() for r in probes]
        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return st

    ref = run("ref", torch.float32, True, start)
    unmerged = run("unmerged", torch.bfloat16, True, start)
    merged = run("merged", torch.bfloat16, False, args.merged)

    def vs(a: dict, b: dict) -> dict:
        flips = a["argmax"] != b["argmax"]
        return {
            "top1_agreement": round(1 - float(flips.float().mean()), 6),
            "flips": int(flips.sum()),
            "median_ref_margin_at_flips": round(float(ref["margin"][flips].median()), 4)
            if flips.any()
            else None,
            "mean_abs_lp_diff": round(float((a["lp"] - b["lp"]).abs().mean()), 6),
            "max_abs_lp_diff": round(float((a["lp"] - b["lp"]).abs().max()), 4),
        }

    diag = {
        "start": start,
        "adapter": args.adapter,
        "merged": args.merged,
        "positions": int(ref["argmax"].numel()),
        "ref_margin_median": round(float(ref["margin"].median()), 4),
        "ref_margin_share_below_0.25": round(float((ref["margin"] < 0.25).float().mean()), 4),
        "unmerged_bf16_vs_fp32_ref": vs(unmerged, ref),
        "merged_bf16_vs_fp32_ref": vs(merged, ref),
        "merged_vs_unmerged_bf16": vs(merged, unmerged),
        "nll_fp32_ref": round(float(-ref["lp"].mean()), 6),
        "nll_unmerged_bf16": round(float(-unmerged["lp"].mean()), 6),
        "nll_merged_bf16": round(float(-merged["lp"].mean()), 6),
    }
    u, m, st = logits["unmerged"], logits["merged"], logits["start"]
    n = sum(x.shape[0] for x in u)
    err = [(a - b).abs() for a, b in zip(m, u)]
    eff = [(a - b).abs() for a, b in zip(u, st)]
    probe = {
        "probe_positions": n,
        "max_abs_merge_error": round(max(float(e.max()) for e in err), 4),
        "max_abs_adapter_effect": round(max(float(e.max()) for e in eff), 4),
        "mean_abs_merge_error": round(sum(float(e.mean()) * e.shape[0] for e in err) / n, 5),
        "mean_abs_adapter_effect": round(sum(float(e.mean()) * e.shape[0] for e in eff) / n, 5),
    }
    probe["ratio"] = round(probe["max_abs_merge_error"] / probe["max_abs_adapter_effect"], 4)
    probe["ratio_mean"] = round(probe["mean_abs_merge_error"] / probe["mean_abs_adapter_effect"], 4)
    probe["ratio_flag"] = probe["ratio"] > RATIO_FLAG
    return diag, probe


def diagnose(args) -> dict:
    diag, _ = measure(args)
    diag["merge_adds_flips"] = (
        diag["merged_bf16_vs_fp32_ref"]["flips"] - diag["unmerged_bf16_vs_fp32_ref"]["flips"]
    )
    return {"check": "merge_diagnose", **diag, "passed": True}  # a measurement, not a gate


def check(args) -> dict:
    diag, probe = measure(args)
    return {
        "check": "merge",
        "gate": "amended 2026-10-06",
        **gate(diag),
        "measurement": diag,
        "probes": probe,
        "checkpoint_sha256": digest(args.merged),
    }


def regate(args) -> dict:
    """The amended gate from a recorded diagnose run and a digest json, no model loaded."""
    run = Path(args.run_dir)
    diag = json.loads((run / "merge_diagnose.json").read_text())
    first = run / "merge_check.json"
    original = run / "merge_check_original.json"
    if first.exists() and not original.exists():
        first.rename(original)
    old = json.loads(original.read_text())
    probe = {
        k: old.get(k)
        for k in (
            "probe_positions",
            "max_abs_merge_error",
            "max_abs_adapter_effect",
            "mean_abs_merge_error",
            "mean_abs_adapter_effect",
            "ratio",
            "ratio_mean",
            "ratio_flag",
        )
    }
    return {
        "check": "merge",
        "gate": "amended 2026-10-06 (applied to the diagnose measurement)",
        **gate(diag),
        "first_gate": {
            "top1_agreement_3_probes": old.get("top1_agreement"),
            "passed": old.get("passed"),
        },
        "measurement": diag,
        "probes": probe,
        "checkpoint_sha256": json.loads((run / "checkpoint_sha256.json").read_text()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("noop")
    a.add_argument("--start", required=True)
    a.add_argument("--config", required=True, help="the stage config whose lora section is used")
    a.add_argument("--work", required=True, help="scratch dir, deleted afterwards")
    a.add_argument("--out", required=True)
    for name in ("check", "diagnose"):
        b = sub.add_parser(name)
        b.add_argument("--adapter", required=True, help="adapter dir: the run root or checkpoint-N")
        b.add_argument("--merged", required=True)
        b.add_argument("--val", required=True)
        b.add_argument("--out", required=True)
    c = sub.add_parser("regate")
    c.add_argument("--run-dir", required=True, help="results/runs/<run> with merge_diagnose.json")
    c.add_argument("--out", required=True)
    d = sub.add_parser("digest")
    d.add_argument("--merged", required=True)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "digest":
        res = {**digest(args.merged), "passed": True}
    else:
        res = {"noop": noop, "check": check, "diagnose": diagnose, "regate": regate}[args.cmd](args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))
    if not res["passed"]:
        raise SystemExit(f"{res['check']} check failed: {out}")
    if (res.get("probes") or {}).get("ratio_flag"):
        print(
            f"FLAG: probe max merge error / adapter effect = {res['probes']['ratio']} (> {RATIO_FLAG})"
        )


if __name__ == "__main__":
    main()
