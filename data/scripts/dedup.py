"""Near-duplicate removal with MinHash LSH over word 5-gram shingles.

Keeps the first document seen in each duplicate cluster. Run this BEFORE pack.py's
train/val split so near-duplicates can't leak across the split.
"""

import argparse
import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

TOKEN = re.compile(r"\w+")


def minhash(text: str, n: int, num_perm: int) -> MinHash:
    words = TOKEN.findall(text.lower())
    m = MinHash(num_perm=num_perm)
    for i in range(max(1, len(words) - n + 1)):
        m.update(" ".join(words[i : i + n]).encode())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/interim/filtered.jsonl")
    ap.add_argument("--out", default="data/processed/dedup.jsonl")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--ngram", type=int, default=5)
    args = ap.parse_args()

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0

    with open(args.inp) as fin, out.open("w") as fout:
        for line in tqdm(fin, desc="dedup"):
            doc = json.loads(line)
            m = minhash(doc["text"], args.ngram, args.num_perm)
            if lsh.query(m):
                dropped += 1
                continue
            lsh.insert(doc["chunk_id"], m)
            fout.write(line)
            kept += 1

    print(f"kept {kept}, dropped {dropped} ({dropped / max(1, kept + dropped):.1%}) -> {out}")


if __name__ == "__main__":
    main()
