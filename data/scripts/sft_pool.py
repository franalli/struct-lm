"""A1: the chunk pool for SFT synthesis -> data/sft/work/pool.jsonl

  .venv/bin/python data/scripts/sft_pool.py               # ~900 ordinary chunks + the seen half
  .venv/bin/python data/scripts/sft_pool.py --pilot 10    # the first 10 of each, same order

forced    every chunk in eval/tasks/sft_seen_chunks.txt (origin eval_seen): what makes the seen
          half of domain_qa / vocab seen by construction
ordinary  --ordinary chunks from the allowed free pool (sft_guard: outside eval_chunk_ids.txt,
          seen-hash, not next to an unseen eval chunk), eligible as in make_tasks (250-600 tokens,
          > 60% letters, prose) with at least 3 numbers, so the passage has values to ask about.
          Each document's share is proportional to its eligible tokens (largest remainder).

Every ordinary chunk is assigned its wording before any question is written: "exact" chunks get a
neutral asker and, at assembly, the eval's own instruction text; "paraphrased" chunks get one of
the 12 personas and a paraphrased instruction. Forced chunks are assigned per fact later (phrasing
1 exact, phrasing 2 paraphrased). Orders and assignments hash the chunk id, so the pilot pool is
the head of the full one and keeps its assignments.
"""

import argparse
import re
from collections import Counter

from sft_common import (
    NEUTRAL_ASKER,
    PERSONAS,
    WORK,
    doc_titles,
    h01,
    load_chunks,
    make_tasks,
    pick,
    quotas,
    update_stats,
    write_jsonl,
)
from sft_guard import Guard

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinary", type=int, default=900)
    ap.add_argument("--pilot", type=int, default=0, help="keep the first N forced + N ordinary")
    args = ap.parse_args()

    mt = make_tasks()
    guard = Guard()
    by_id, _ = load_chunks()
    titles = doc_titles()

    off = Counter(guard.why_not(c) for c in by_id if not guard.allowed(c))
    forced = sorted(guard.seen, key=lambda c: h01(f"sft-pool:{c}"))
    missing = [c for c in forced if c not in by_id]
    assert not missing, f"seen chunks missing from chunks.jsonl: {missing[:3]}"

    eligible = [
        c
        for cid, c in by_id.items()
        if cid not in guard.seen
        and guard.allowed(cid)
        and 250 <= c["n_tokens"] <= 600
        and mt.alpha_ratio(c["text"]) > 0.6
        and not mt.is_nonprose(c["text"])
        and len(NUMBER.findall(c["text"])) >= 3
    ]
    weights = Counter()
    for c in eligible:
        weights[c["doc"]] += c["n_tokens"]
    quota = quotas(weights, args.ordinary)
    ordinary = []
    for doc, n in quota.items():
        cands = sorted(
            (c for c in eligible if c["doc"] == doc), key=lambda c: h01(f"sft-pool:{c['chunk_id']}")
        )
        ordinary += [c["chunk_id"] for c in cands[:n]]
    ordinary.sort(key=lambda c: h01(f"sft-pool:{c}"))
    if args.pilot:
        forced, ordinary = forced[: args.pilot], ordinary[: args.pilot]

    def row(cid: str, origin: str) -> dict:
        c = by_id[cid]
        exact = origin == "ordinary" and h01(f"sft-wording:{cid}") < 0.5
        wording = None if origin == "eval_seen" else ("exact" if exact else "paraphrased")
        persona = pick(f"sft-persona:{cid}", PERSONAS) if wording == "paraphrased" else None
        return {
            "chunk_id": cid,
            "doc": c["doc"],
            "title": titles.get(c["doc"], c["title"]),
            "text": c["text"],
            "n_tokens": c["n_tokens"],
            "origin": origin,
            "wording": wording,
            "persona": persona,
            "asker": persona or NEUTRAL_ASKER,
        }

    pool = [row(c, "eval_seen") for c in forced] + [row(c, "ordinary") for c in ordinary]
    write_jsonl(WORK / "pool.jsonl", pool)
    stats = {
        "chunks": len(by_id),
        "allowed": len(by_id) - sum(off.values()),
        "off_limits": dict(off),
        "buffer_pages": len(guard.buffer_pages),
        "eligible_ordinary": len(eligible),
        "eligible_docs": len(weights),
        "forced": len(forced),
        "ordinary": len(ordinary),
        "ordinary_docs": len({by_id[c]["doc"] for c in ordinary}),
        "wording": dict(Counter(r["wording"] for r in pool if r["origin"] == "ordinary")),
        "pilot": args.pilot,
    }
    update_stats("pool", stats)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
