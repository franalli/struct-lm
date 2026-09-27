"""Exact and near-duplicate removal over the whole corpus, first copy wins (sources.csv order).

Rewrites data/processed/docs.jsonl in place (so rerun from filter.py, not on its own output) and
writes data/processed/duplicates.jsonl: every near-duplicate cluster found, most copies first.

Exact: sha1 of the normalised document text (lowercase alphanumeric words), which catches the
same manual fetched from two URLs and byte-identical re-issues.
Near: MinHash LSH over paragraphs (5-word shingles, 128 permutations, Jaccard >= 0.8), across
documents and within them. USACE manuals share boilerplate paragraphs (distribution
statements, purpose sections, definitions) and FHWA manuals repeat AASHTO clauses verbatim;
paragraph level removes those while keeping the rest of each document. Paragraphs under 5 words
have no shingle and are left alone. A document left under MIN_WORDS (2,000) words is dropped, as
filter.py does before dedup (a later edition can shrink to a stub), and its paragraphs are taken
back out of the index so later copies of them survive.

The seed (eval) documents come first in sources.csv, and this runs before split.py, so no
near-duplicate paragraph can end up in both train and val.
"""

import argparse
import hashlib
import re

from common import BASE, MIN_WORDS, load_tokenizer, read_jsonl, update_stats, write_jsonl
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def minhash(ws: list[str], n: int, template: MinHash) -> MinHash:
    # reuse the template's permutations: building 128 of them per paragraph dominated the runtime
    m = MinHash(len(template), permutations=template.permutations, scheme=template.scheme)
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
    template = MinHash(num_perm=args.num_perm)
    origin: dict[str, tuple[str, str]] = {}  # lsh key -> (slug, paragraph) of the kept copy
    clusters: dict[str, dict] = {}  # lsh key -> cluster, created at its first removed copy
    n_paras = 0
    out, stubs = [], []
    for di, d in enumerate(tqdm(unique, desc="near-dup")):
        kept, inserted = [], []
        for pi, para in enumerate(d["text"].split("\n\n")):
            n_paras += 1
            ws = words(para)
            if len(ws) < args.ngram:
                kept.append(para)
                continue
            m = minhash(ws, args.ngram, template)
            if match := lsh.query(m):
                key = min(match)  # keys sort in corpus order: the earliest kept copy
                slug, text = origin[key]
                c = clusters.setdefault(
                    key, {"text": text, "first_in": slug, "copies_removed": 0, "docs": {slug}}
                )
                c["copies_removed"] += 1
                c["docs"].add(d["slug"])
                continue
            key = f"{di:05d}:{pi:06d}"
            lsh.insert(key, m)
            origin[key] = (d["slug"], para)
            inserted.append(key)
            kept.append(para)
        text = "\n\n".join(kept)
        if len(text.split()) < MIN_WORDS:
            stubs.append({"slug": d["slug"], "words": len(text.split())})
            for key in inserted:  # this document won't be kept, so it can't be the first copy
                lsh.remove(key)
                clusters.pop(key, None)  # copies removed from this same document only
            continue
        out.append({**d, "text": text, "n_tokens": len(encode(text))})
    write_jsonl(args.docs, out)

    # most copies first; ties keep first-removal order
    clusters = [
        {
            "text": c["text"],
            "first_in": c["first_in"],
            "copies_removed": c["copies_removed"],
            "docs_with_copy": len(c["docs"]),
            "docs": sorted(c["docs"]),
        }
        for c in sorted(clusters.values(), key=lambda c: -c["copies_removed"])
    ]
    write_jsonl(args.clusters, clusters)
    n_near = sum(c["copies_removed"] for c in clusters)

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
            "docs_dropped_short": stubs,  # under MIN_WORDS words once duplicates were removed
            "tokens_in": tokens_in,
            "tokens_after_exact": tokens_exact,
            "tokens_out": tokens_out,
            "token_reduction": round(1 - tokens_out / tokens_in, 4),
            "top_duplicated": top,
            "top_cross_document": top_docs,
        },
    )
    print(f"exact duplicates (dropped, kept): {exact}")
    print(f"dropped as under {MIN_WORDS:,} words after dedup: {stubs}")
    print(f"near-dup paragraphs: {n_near:,} of {n_paras:,} in {len(clusters):,} clusters")
    print(
        f"tokens {tokens_in:,} -> {tokens_exact:,} (exact) -> {tokens_out:,} (near) "
        f"= {1 - tokens_out / tokens_in:.1%} removed -> {args.docs}, clusters -> {args.clusters}"
    )
    for t in top + top_docs:
        print(f"  {t['copies_removed']:>4}x in {t['docs_with_copy']:>3} docs  {t['text'][:110]!r}")


if __name__ == "__main__":
    main()
