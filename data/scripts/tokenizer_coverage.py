"""How well does each candidate tokenizer fit the domain corpus?

Reports fertility (tokens per whitespace word), byte-fallback / UNK rate, and the most
fragmented domain terms. High fertility on key terms means CPT has more to learn, and
may justify vocabulary extension; record the verdict in notes/decisions.md.
"""

import argparse
import json
import random
import re
from collections import Counter

from transformers import AutoTokenizer

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/dedup.jsonl")
    ap.add_argument("--tokenizers", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=500, help="documents to sample")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    with open(args.inp) as f:
        docs = [json.loads(line)["text"] for line in f]
    random.Random(0).shuffle(docs)
    docs = docs[: args.sample]
    vocab = Counter(w for d in docs for w in WORD.findall(d))

    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(name)
        n_words = n_tokens = n_unk = 0
        for d in docs:
            ids = tok(d, add_special_tokens=False)["input_ids"]
            n_words += len(d.split())
            n_tokens += len(ids)
            if tok.unk_token_id is not None:
                n_unk += ids.count(tok.unk_token_id)
        frag = sorted(
            ((len(tok.tokenize(" " + w)), w) for w, _ in vocab.most_common(2000)), reverse=True
        )
        print(f"\n== {name}")
        print(f"fertility   {n_tokens / n_words:.3f} tokens/word")
        print(f"unk rate    {n_unk / max(1, n_tokens):.5f}")
        print(f"most fragmented frequent terms (top {args.top}):")
        for n, w in frag[: args.top]:
            print(f"  {n:>3}  {w}")


if __name__ == "__main__":
    main()
