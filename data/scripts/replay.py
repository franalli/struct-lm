"""FineWeb-Edu slices for Stage 2: data/processed/replay.jsonl and general_val.jsonl.

replay.jsonl (the replay ablation): streams HuggingFaceFW/fineweb-edu (config sample-10BT) in its
fixed order and takes documents until the slice holds 10% of the train split's Tekken tokens.
Text, not pre-tokenised shards (those use other tokenizers). tokenizer_coverage.py reuses the
first ~1M tokens as its general-English comparison sample.

general_val.jsonl (~500k tokens, the ppl_general_val column): the head of the LAST parquet shard
of sample-10BT, not the tokens after the replay slice, so it stays the same file when replay grows
with the corpus (replay at 80M train tokens is still well inside the first shard). Disjoint from
replay by construction; the id check below makes sure.
"""

import argparse
import os
import sys
import traceback

from common import BASE, load_tokenizer, read_jsonl, update_stats, write_jsonl
from datasets import load_dataset
from split import SEQ_LEN, packed

GENERAL_SHARD = "sample/10BT/013_00000.parquet"  # the last of sample-10BT's 14 shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--out", default="data/processed/replay.jsonl")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--general-out", default="data/processed/general_val.jsonl")
    ap.add_argument("--general-tokens", type=int, default=500_000)
    args = ap.parse_args()

    train = read_jsonl(args.train)
    target = int(args.frac * sum(d["n_tokens"] for d in train))
    encode, _ = load_tokenizer(BASE)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    rows, tokens = take(stream, encode, target)
    write_jsonl(args.out, rows)

    general_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files=GENERAL_SHARD,
        split="train",
        streaming=True,
    )
    general, general_tokens = take(general_stream, encode, args.general_tokens)
    overlap = {r["id"] for r in rows} & {r["id"] for r in general}
    if overlap:
        raise SystemExit(f"general_val shares {len(overlap)} documents with replay")
    write_jsonl(args.general_out, general)

    update_stats(
        "replay",
        {
            "source": "HuggingFaceFW/fineweb-edu sample-10BT (stream order)",
            "target_tokens": target,
            "docs": len(rows),
            "tokens": tokens,
            "packed_seqs": packed(rows),
            "packed_seqs_train_plus_replay": packed(train + rows),
            "general_val": {
                "source": f"HuggingFaceFW/fineweb-edu {GENERAL_SHARD} (stream order)",
                "docs": len(general),
                "tokens": general_tokens,
                "packed_seqs": packed(general),
            },
        },
    )
    print(
        f"{len(rows):,} docs, {tokens:,} tokens (target {target:,}) -> {args.out}; "
        f"train+replay packs into {packed(train + rows):,} x {SEQ_LEN}"
    )
    print(f"{len(general):,} docs, {general_tokens:,} tokens -> {args.general_out}")


def take(stream, encode, target: int) -> tuple[list[dict], int]:
    """Documents from the head of a stream until they hold at least `target` tokens."""
    rows, tokens = [], 0
    for ex in stream:
        n = len(encode(ex["text"]))
        rows.append({"id": ex["id"], "url": ex["url"], "text": ex["text"], "n_tokens": n})
        tokens += n
        if tokens >= target:
            break
    return rows, tokens


if __name__ == "__main__":
    # Leaving the stream early leaves a parquet read-ahead task in pyarrow's thread pool; at
    # C-level exit the pool's destructor waits for it, but it needs the (finalized) interpreter
    # to read via fsspec, so a normal exit deadlocks. Exit without the C-level teardown, on
    # failure too (with status 1), once output and tracebacks are flushed.
    code = 0
    try:
        main()
    except Exception:  # noqa: BLE001  any failure must still take the os._exit path below
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
