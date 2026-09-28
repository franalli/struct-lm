"""Document-level train/val split: docs.jsonl -> data/processed/train.jsonl, val.jsonl.

Whole documents go to one side, never paragraphs, so val perplexity measures whether CPT
generalises to unseen documents of the same kind (the KPI eval measures whether it absorbed the
ones it saw). Per publisher, max(1, round(5% of its documents)) go to val, so every source has a
held-out document; they are picked by sha256(slug), which is stable across runs. Every document
the eval samples from (eval/tasks/eval_docs.txt) stays in train.

No tokenising or packing here: train/packing.py packs at load time (each document as
BOS + tokens + EOS, concatenated, cut into fixed 4,096-token windows, the last partial window
dropped). This reports that window count, and the effective batch the 150-step rule gives it, so
Stage 2 can size its schedule: one epoch of about 150 optimizer steps, the batch a power of two
in sequences (32 at ~20M tokens, 64 at ~40M, 128 at ~80M; notes/decisions.md).
"""

import argparse
import hashlib
import math
from collections import defaultdict

from common import eval_docs, read_jsonl, update_stats, write_jsonl

SEQ_LEN = 4096
TARGET_STEPS = 150  # optimizer steps per epoch: enough for LoRA to settle, a sane LR schedule


def packed(docs: list[dict]) -> int:
    return sum(d["n_tokens"] + 2 for d in docs) // SEQ_LEN  # +2: BOS and EOS around each document


def seqs_per_step(n_seqs: int) -> int:
    """Effective batch in sequences: the power of two closest to one epoch in TARGET_STEPS."""
    return 2 ** round(math.log2(n_seqs / TARGET_STEPS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/docs.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    docs = read_jsonl(args.inp)
    held = eval_docs()
    missing = held - {d["slug"] for d in docs}
    if missing:
        raise SystemExit(f"eval documents missing from {args.inp}: {sorted(missing)}")

    by_pub: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_pub[d["publisher"]].append(d)
    val_slugs, no_val = set(), []
    for pub, ds in by_pub.items():
        candidates = sorted(
            (d["slug"] for d in ds if d["slug"] not in held),
            key=lambda s: hashlib.sha256(s.encode()).hexdigest(),
        )
        n_val = max(1, round(args.val_frac * len(ds)))
        val_slugs.update(candidates[:n_val])
        if not candidates:
            no_val.append(pub)

    train = [d for d in docs if d["slug"] not in val_slugs]
    val = [d for d in docs if d["slug"] in val_slugs]
    write_jsonl("data/processed/train.jsonl", train)
    write_jsonl("data/processed/val.jsonl", val)

    def side(ds: list[dict]) -> dict:
        return {"docs": len(ds), "tokens": sum(d["n_tokens"] for d in ds)}

    tokens = sum(d["n_tokens"] for d in docs)
    stats = {
        "by_publisher": {
            pub: {
                "train": side([d for d in ds if d["slug"] not in val_slugs]),
                "val": side([d for d in ds if d["slug"] in val_slugs]),
            }
            for pub, ds in sorted(by_pub.items())
        },
        "train": side(train),
        "val": side(val),
        "val_docs": sorted(val_slugs),
        "val_token_frac": round(side(val)["tokens"] / tokens, 4),
        "eval_docs_in_train": len(held),
        "publishers_without_val": no_val,
        "seq_len": SEQ_LEN,
        "packed_seqs": {"train": packed(train), "val": packed(val)},
        "seqs_per_step": seqs_per_step(packed(train)),
        "steps_per_epoch": math.ceil(packed(train) / seqs_per_step(packed(train))),
    }
    update_stats("split", stats)
    for pub, s in stats["by_publisher"].items():
        print(
            f"{pub:<6} train {s['train']['docs']:>4} docs {s['train']['tokens']:>11,}   "
            f"val {s['val']['docs']:>3} docs {s['val']['tokens']:>10,}"
        )
    print(
        f"train {side(train)}, val {side(val)} ({stats['val_token_frac']:.1%} of tokens); "
        f"{len(held)} eval docs held in train"
    )
    print(
        f"packed at {SEQ_LEN}: {stats['packed_seqs']}; 150-step rule: "
        f"{stats['seqs_per_step']} seqs/step, {stats['steps_per_epoch']} optimizer steps/epoch"
    )
    if no_val:
        print(f"WARNING: no held-out document possible for {no_val} (all their docs are eval docs)")


if __name__ == "__main__":
    main()
