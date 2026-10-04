"""Paired comparison of two runs' perplexity, with bootstrap confidence intervals.

  .venv/bin/python eval/ppl_compare.py base-8b cpt-8b            # b vs a, every slice
  .venv/bin/python eval/ppl_compare.py cpt-8b cpt-8b-seed1 --set domain_val

Reads results/ppl/<run>.json (eval/perplexity.py). Every run is scored on the same windows, so the
comparison is paired: each bootstrap resample draws the same windows (or documents) for both runs
and recomputes the difference in mean NLL per predicted token. Two resampling units:
  windows    the 4,096-token blocks (296 in domain val, 122 in general val): tight, but the
             windows of one document are correlated, so it understates the uncertainty
  documents  whole documents (12 in domain val, 512 in general val): the honest unit for "would
             this hold on other documents", and wide when there are few of them
Prints delta nats per token (b - a) and the perplexity change exp(delta) - 1, each with a 95%
percentile interval. A delta whose interval covers 0 is within sampling noise; the seed-to-seed
delta (cpt-8b vs cpt-8b-seed1) is the run-to-run noise floor the ablation deltas are read against.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

SETS = ("domain_val", "general_val", "train_slice", "postcutoff")


def sums(res: dict, set_name: str, unit: str) -> dict[str, tuple[float, float]]:
    """{key: (summed NLL, predicted tokens)} for one slice, keyed by window index or document."""
    if unit == "windows":
        # .get: postcutoff exists only for runs measured after 2026-10-04
        return {str(i): tuple(w) for i, w in enumerate(res["sums"]["windows"].get(set_name, []))}
    return {k: tuple(v) for k, v in res["sums"]["docs"].get(set_name, {}).items()}


def compare(a: dict, b: dict, set_name: str, unit: str, n: int = 10_000, seed: int = 0) -> dict:
    sa, sb = sums(a, set_name, unit), sums(b, set_name, unit)
    keys = sorted(set(sa) & set(sb))
    if not keys:
        return {}
    A = np.array([sa[k] for k in keys])
    B = np.array([sb[k] for k in keys])
    if not np.array_equal(A[:, 1], B[:, 1]):
        raise SystemExit(f"{set_name}/{unit}: token counts differ; not the same windows")

    def delta(idx: np.ndarray) -> np.ndarray:
        a_, b_ = A[idx].sum(axis=-2), B[idx].sum(axis=-2)
        return b_[..., 0] / b_[..., 1] - a_[..., 0] / a_[..., 1]

    point = float(delta(np.arange(len(keys))))
    boot = delta(np.random.default_rng(seed).integers(0, len(keys), (n, len(keys))))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "units": len(keys),
        "nats": point,
        "nats_ci": (float(lo), float(hi)),
        "ppl_pct": (math.exp(point) - 1) * 100,
        "ppl_pct_ci": ((math.exp(lo) - 1) * 100, (math.exp(hi) - 1) * 100),
    }


def fmt(c: dict) -> str:
    (lo, hi), (plo, phi) = c["nats_ci"], c["ppl_pct_ci"]
    return (
        f"{c['nats']:+.4f} nats [{lo:+.4f}, {hi:+.4f}]  ppl {c['ppl_pct']:+.2f}% "
        f"[{plo:+.2f}, {phi:+.2f}]  ({c['units']} units)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="reference run (e.g. base-8b)")
    ap.add_argument("b", help="compared run")
    ap.add_argument("--set", choices=SETS, action="append", help="default: all three")
    ap.add_argument("--dir", default="results/ppl")
    ap.add_argument("--n", type=int, default=10_000, help="bootstrap resamples")
    args = ap.parse_args()
    a, b = (json.loads((Path(args.dir) / f"{r}.json").read_text()) for r in (args.a, args.b))
    for r, res in ((args.a, a), (args.b, b)):
        if "sums" not in res:
            raise SystemExit(f"{r}: no per-window sums; rerun eval/perplexity.py for it")
    print(f"{args.b} vs {args.a} (b - a; 95% bootstrap intervals, paired)")
    for set_name in args.set or SETS:
        for unit in ("windows", "documents"):
            if c := compare(a, b, set_name, unit, args.n):
                print(f"  {set_name:<12} {unit:<9} {fmt(c)}")


if __name__ == "__main__":
    main()
