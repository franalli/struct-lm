"""Is the corpus already in the base model's pretraining data? A first, cheap check: per-document base
perplexity against the document's date. If the model had seen the older documents and not the newer
ones, newer documents should be harder for it, within the same publisher.

  .venv/bin/python eval/exposure_check.py [--ppl results/ppl/base-8b.json]

Per-document base perplexity from results/ppl/<run>.json, no GPU: the 12 val documents exactly
(by_doc), plus every train-slice window that one document fills to >= 90% (a 4,096-token sample
of that document). Dates: the PDF's creationDate (data/raw/<slug>.pdf), which is not always the
publication date. Perplexity is taken relative to the publisher's median, since publishers differ
in difficulty (USACE manuals are hardest); the Spearman correlation with year gets a
permutation p-value.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from packing import pack, spread

BASE = "mistralai/Ministral-3-8B-Base-2512"


def pdf_year(slug: str) -> int | None:
    import pymupdf

    with pymupdf.open(f"data/raw/{slug}.pdf") as d:
        y = (d.metadata or {}).get("creationDate", "")[2:6]
    return int(y) if y.isdigit() else None


def spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 10_000) -> tuple[float, float]:
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(perm) >= abs(rho)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl", default="results/ppl/base-8b.json")
    args = ap.parse_args()
    base = json.loads(Path(args.ppl).read_text())

    docs = {s: (v["publisher"], v["ppl"], "val") for s, v in base["by_doc"].items()}
    tr = pack(["data/processed/train.jsonl"], BASE)
    for k, i in enumerate(spread(len(tr.ids))):
        ids, counts = np.unique(tr.doc[i, 1:], return_counts=True)
        j = counts.argmax()
        if counts[j] / counts.sum() >= 0.9:
            nll, n = base["sums"]["windows"]["train_slice"][k]
            slug = tr.doc_names[ids[j]]
            pub = tr.sources[tr.doc_source[ids[j]]]
            docs.setdefault(slug, (pub, math.exp(nll / n), "train window"))

    by_pub = defaultdict(list)
    for slug, (pub, ppl, src) in docs.items():
        if (y := pdf_year(slug)) is not None:
            by_pub[pub].append((slug, y, ppl, src))
    rows = [
        (y, ppl / np.median([r[2] for r in rs]) - 1, slug, pub, ppl, src)
        for pub, rs in by_pub.items()
        for slug, y, ppl, src in rs
    ]
    years, rel = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    rho, p = spearman(years, rel)
    n_val = sum(v[2] == "val" for v in docs.values())
    print(f"{len(docs)} documents with a base perplexity ({n_val} val), {len(rows)} dated")
    print(f"Spearman(year, perplexity relative to publisher median) = {rho:+.2f}, p = {p:.2f}")
    for lo, hi in ((1990, 2014), (2015, 2019), (2020, 2023), (2024, 2026)):
        sel = rel[(years >= lo) & (years <= hi)]
        if len(sel):
            print(f"  {lo}-{hi}: n={len(sel):2d}  median {np.median(sel) * 100:+5.1f}%")
    print("2024 and later:")
    for y, r, slug, pub, ppl, src in sorted(rows, reverse=True):
        if y >= 2024:
            print(f"  {y} {slug:42} {pub:6} {ppl:6.3f} ({r * 100:+.1f}% vs publisher) [{src}]")


if __name__ == "__main__":
    main()
