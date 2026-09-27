"""Train/val split BY DOCUMENT, then tokenize and pack into fixed-length sequences.

Splitting by document (a stable hash of the doc id) rather than by packed chunk keeps
any one document entirely on one side of the split, so val loss is honest.
"""

import argparse
import hashlib
import json

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def is_val(doc_id: str, val_frac: float) -> bool:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < val_frac


def pack(texts: list[str], tok, seq_len: int) -> dict[str, list[list[int]]]:
    buf: list[int] = []
    chunks: list[list[int]] = []
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)  # document boundary
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
    return {"input_ids": chunks}  # the tail < seq_len is dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/dedup.jsonl")
    ap.add_argument("--out", default="data/packed/cpt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Records may be page chunks ({"doc": slug, ...}); rejoin them per document, in file
    # order, so the split is by document and packing sees contiguous text.
    docs: dict[str, list[str]] = {}
    with open(args.inp) as f:
        for line in f:
            rec = json.loads(line)
            docs.setdefault(rec["doc"], []).append(rec["text"])
    splits: dict[str, list[str]] = {"train": [], "validation": []}
    for doc_id, texts in docs.items():
        splits["validation" if is_val(doc_id, args.val_frac) else "train"].append(
            "\n\n".join(texts)
        )

    ds = DatasetDict({k: Dataset.from_dict(pack(v, tok, args.seq_len)) for k, v in splits.items()})
    ds.save_to_disk(args.out)
    for k, v in ds.items():
        print(
            f"{k:<10} {len(splits[k]):>7} docs  {len(v):>8} seqs  "
            f"{len(v) * args.seq_len / 1e6:,.1f}M tokens"
        )


if __name__ == "__main__":
    main()
