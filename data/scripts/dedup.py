"""Exact and near-duplicate removal over the whole corpus, first copy wins (sources.csv order).

Rewrites data/processed/docs.jsonl in place (so rerun from filter.py, not on its own output) and
writes data/processed/duplicates.jsonl: every near-duplicate cluster found, most copies first.

Exact: sha1 of the normalised document text (lowercase alphanumeric words), which catches the
same manual fetched from two URLs and byte-identical re-issues.
Near: MinHash LSH over paragraphs (5-word shingles, 128 permutations, Jaccard >= 0.8), across
documents and within them. USACE manuals share boilerplate paragraphs (distribution
statements, purpose sections, definitions) and FHWA manuals repeat AASHTO clauses verbatim;
paragraph level removes those while keeping the rest of each document. Paragraphs under 5 words
have no shingle and are left alone.

The seed (eval) documents come first in sources.csv, and this runs before split.py, so no
near-duplicate paragraph can end up in both train and val.
"""

import argparse
import hashlib
import re
from collections import Counter

from common import BASE, load_tokenizer, read_jsonl, update_stats, write_jsonl
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def minhash(ws: list[str], n: int, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    m.update_batch([" ".join(ws[i : i + n]).encode() for i in range(len(ws) - n + 1)])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/processed/docs.jsonl")
    ap.add_argument("--clusters", default="data/processed/duplicates.jsonl")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard similarity")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    encode, _ = load_tokenizer(BASE)
    docs = read_jsonl(args.docs)

    first: dict[str, str] = {}
    exact, unique = [], []
    for d in docs:
        h = hashlib.sha1(" ".join(words(d["text"])).encode()).hexdigest()
        if h in first:
            exact.append([d["slug"], first[h]])
        else:
            first[h] = d["slug"]
            unique.append(d)

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    origin: dict[str, tuple[str, str]] = {}  # lsh key -> (slug, paragraph) of the kept copy
    removed: Counter[str] = Counter()  # lsh key -> copies removed
    copy_docs: dict[str, set[str]] = {}
    n_paras = n_near = 0
    out = []
    for di, d in enumerate(tqdm(unique, desc="near-dup")):
        kept = []
        for pi, para in enumerate(d["text"].split("\n\n")):
            n_paras += 1
            ws = words(para)
            if len(ws) < args.ngram:
                kept.append(para)
                continue
            m = minhash(ws, args.ngram, args.num_perm)
            if match := lsh.query(m):
                key = min(match)  # keys sort in corpus order: the earliest kept copy
                removed[key] += 1
                copy_docs.setdefault(key, {origin[key][0]}).add(d["slug"])
                n_near += 1
                continue
            key = f"{di:05d}:{pi:06d}"
            lsh.insert(key, m)
            origin[key] = (d["slug"], para)
            kept.append(para)
        if kept:
            text = "\n\n".join(kept)
            out.append({**d, "text": text, "n_tokens": len(encode(text))})
    write_jsonl(args.docs, out)

    def cluster(k: str) -> dict:
        return {
            "text": origin[k][1],
            "first_in": origin[k][0],
            "copies_removed": removed[k],
            "docs_with_copy": len(copy_docs[k]),
            "docs": sorted(copy_docs[k]),
        }

    clusters = [cluster(k) for k, _ in removed.most_common()]
    write_jsonl(args.clusters, clusters)

    def short(c: dict) -> dict:
        return {k: (v[:300] if k == "text" else v) for k, v in c.items() if k != "docs"}

    # by copies: dominated by per-chapter running headers (each on < 30% of a long manual's
    # pages, so extract.py keeps them); by documents: boilerplate shared across publications
    top = [short(c) for c in clusters[: args.top]]
    top_docs = [short(c) for c in sorted(clusters, key=lambda c: -c["docs_with_copy"])[: args.top]]
    tokens_in = sum(d["n_tokens"] for d in docs)
    tokens_exact = sum(d["n_tokens"] for d in unique)
    tokens_out = sum(d["n_tokens"] for d in out)
    update_stats(
        "dedup",
        {
            "docs_in": len(docs),
            "exact_duplicates": exact,  # [dropped, kept]
            "docs_out": len(out),
            "paragraphs_in": n_paras,
            "near_dup_paragraphs": n_near,
            "near_dup_clusters": len(clusters),
            "tokens_in": tokens_in,
            "tokens_after_exact": tokens_exact,
            "tokens_out": tokens_out,
            "token_reduction": round(1 - tokens_out / tokens_in, 4),
            "top_duplicated": top,
            "top_cross_document": top_docs,
        },
    )
    print(f"exact duplicates (dropped, kept): {exact}")
    print(f"near-dup paragraphs: {n_near:,} of {n_paras:,} in {len(clusters):,} clusters")
    print(
        f"tokens {tokens_in:,} -> {tokens_exact:,} (exact) -> {tokens_out:,} (near) "
        f"= {1 - tokens_out / tokens_in:.1%} removed -> {args.docs}, clusters -> {args.clusters}"
    )
    for t in top + top_docs:
        print(f"  {t['copies_removed']:>4}x in {t['docs_with_copy']:>3} docs  {t['text'][:110]!r}")


if __name__ == "__main__":
    main()
