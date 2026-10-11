"""Which closed-book answers the text CPT read contains, by half.

  .venv/bin/python eval/cpt_coverage.py     # -> results/cpt_coverage.json

The eval's items were written from data/processed/chunks.jsonl (extract.py --chunks, tables
kept), while CPT read data/processed/train.jsonl, after the cleaning steps (filter.py's
paragraph rules drop table debris, lists and reference entries; dedup.py and pii.py rewrite the
rest). For each domain_qa item, by half (taught = its chunk is in sft_seen_chunks.txt): is its
gold answer in its source chunk word for word (lower case, whitespace collapsed), and, if so, is
it in the CPT text of the same document? An answer not found word for word in its passage
(a paraphrase or a unit change) can't be checked this way and is counted apart.

Both inputs are built by `make data` and not committed: without them this script says so and
leaves the committed results/cpt_coverage.json as it is.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CPT = REPO / "data/processed/train.jsonl"
CHUNKS = REPO / "data/processed/chunks.jsonl"
OUT = REPO / "results/cpt_coverage.json"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def main() -> None:
    if not (CPT.exists() and CHUNKS.exists()):
        sys.exit(f"{CPT.name} or {CHUNKS.name} not built (make data); {OUT.name} left as is")
    cpt = {r["slug"]: norm(r["text"]) for r in map(json.loads, CPT.open())}
    chunks = {r["chunk_id"]: norm(r["text"]) for r in map(json.loads, CHUNKS.open())}
    seen = set((REPO / "eval/tasks/sft_seen_chunks.txt").read_text().split())
    res = {}
    for item in map(json.loads, (REPO / "eval/tasks/domain_qa.jsonl").open()):
        chunk = item["source_chunk"]
        h = res.setdefault(
            "taught" if chunk in seen else "document_only",
            {"items": 0, "in_passage": 0, "not_in_passage": 0, "missing_from_cpt": []},
        )
        h["items"] += 1
        answer = norm(item["answer"])
        if answer not in chunks.get(chunk, ""):
            h["not_in_passage"] += 1
            continue
        h["in_passage"] += 1
        if answer not in cpt.get(chunk.split(":")[0], ""):
            h["missing_from_cpt"].append(item["id"])
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for half, h in res.items():
        print(
            f"{half}: {h['items']} items, {h['in_passage']} answers word for word in their "
            f"passage, {len(h['missing_from_cpt'])} of them missing from the CPT text"
        )


if __name__ == "__main__":
    main()
