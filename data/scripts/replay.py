"""FineWeb-Edu replay slice for the Stage 2 replay ablation: data/processed/replay.jsonl.

Streams HuggingFaceFW/fineweb-edu (config sample-10BT) in its fixed order and takes documents
until the slice holds 10% of the train split's Tekken tokens. Text, not pre-tokenised shards
(those use other tokenizers). tokenizer_coverage.py reuses the first ~1M tokens as its
general-English comparison sample.
"""

import argparse
import os
import sys

from common import BASE, load_tokenizer, read_jsonl, update_stats, write_jsonl
from datasets import load_dataset
from split import SEQ_LEN, packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--out", default="data/processed/replay.jsonl")
    ap.add_argument("--frac", type=float, default=0.10)
    args = ap.parse_args()

    train = read_jsonl(args.train)
    target = int(args.frac * sum(d["n_tokens"] for d in train))
    encode, _ = load_tokenizer(BASE)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    rows, tokens = [], 0
    examples = iter(stream)
    while tokens < target:
        ex = next(examples)
        n = len(encode(ex["text"]))
        rows.append({"id": ex["id"], "url": ex["url"], "text": ex["text"], "n_tokens": n})
        tokens += n
    write_jsonl(args.out, rows)

    update_stats(
        "replay",
        {
            "source": "HuggingFaceFW/fineweb-edu sample-10BT (stream order)",
            "target_tokens": target,
            "docs": len(rows),
            "tokens": tokens,
            "packed_seqs": packed(rows),
            "packed_seqs_train_plus_replay": packed(train + rows),
        },
    )
    print(
        f"{len(rows):,} docs, {tokens:,} tokens (target {target:,}) -> {args.out}; "
        f"train+replay packs into {packed(train + rows):,} x {SEQ_LEN}"
    )


if __name__ == "__main__":
    main()
    # Leaving the stream early leaves a parquet read-ahead task in pyarrow's thread pool; at
    # C-level exit the pool's destructor waits for it, but it needs the (finalized) interpreter
    # to read via fsspec, so a normal exit deadlocks. Everything is written and flushed by now.
    sys.stdout.flush()
    os._exit(0)
