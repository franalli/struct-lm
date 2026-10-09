"""Stage 5 step 0: the GRPO candidate tasks -> data/grpo/tasks.jsonl (+ tasks_meta.json)

  .venv/bin/python data/scripts/grpo_tasks.py

Every closed_book, grounded and abstain prompt of the Stage 4 pool (data/dpo/prompts.jsonl) outside
its `judge` split (the 100 win-rate prompts stay unseen by every trained stage), with the verifier
the reward reads (train/grpo_rewards.verifier_for, from the SFT builder's record in
data/sft/work/judged.jsonl):
  closed_book  {gold, kind, tolerance}: eval/scorers.qa_strict (the strict checker, 2026-10-09)
  grounded     {gold_chunk, chunk_ids, max_distinct}: cite the gold passage, at most 2 passages
  abstain      {phrase, chunk_ids}: exactly the abstain sentence
Definitions are left out (no verifier knows a definition's quality, as in Stage 4).

Rows keep the pool's fields (id, format, prompt, source_chunks, fact_id, dpo_split), so the
contamination check (eval/contamination.py --only grpo) and the probe job (eval/sample.py
grpo_probe) read them as they read the pool. Prompts are the Mistral teacher pool's and the abstain
reply a fixed string: no Claude-written text (rule 13). The probe (grpo_probe.py) then keeps the
tasks the start policy sometimes solves and writes train.jsonl / val.jsonl.
"""

import hashlib
import json
import sys
from collections import Counter

from dpo_common import REPO, records
from sft_common import read_jsonl, write_jsonl

sys.path.append(str(REPO / "train"))  # appended: data/scripts/common.py stays first for `common`
from grpo_rewards import verifier_for

POOL = REPO / "data/dpo/prompts.jsonl"
OUT = REPO / "data/grpo"
FORMATS = ("closed_book", "grounded", "abstain")
KEEP = ("id", "format", "prompt", "source_chunks", "fact_id", "dpo_split")


def main() -> None:
    pool = [r for r in read_jsonl(POOL) if r["format"] in FORMATS and r["dpo_split"] != "judge"]
    recs = records({r["id"] for r in pool})
    rows = []
    for r in pool:
        rec = recs[r["id"]]
        row = {k: r[k] for k in KEEP}
        row["answer_kind"] = rec["kind"] if r["format"] == "closed_book" else None
        row["verifier"] = verifier_for(r["format"], rec)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "tasks.jsonl", rows)
    meta = {
        "source": "data/dpo/prompts.jsonl minus the judge split; closed_book, grounded, abstain",
        "tasks": len(rows),
        "by_format": dict(Counter(r["format"] for r in rows)),
        "by_closed_book_kind": dict(Counter(r["answer_kind"] for r in rows if r["answer_kind"])),
        "closed_book_facts": len({r["fact_id"] for r in rows if r["format"] == "closed_book"}),
        "sha256": hashlib.sha256((OUT / "tasks.jsonl").read_bytes()).hexdigest(),
    }
    (OUT / "tasks_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # the smoke run trains on tasks.jsonl before the probe; grpo_probe.py rewrites this file
    (OUT / "SHA256SUMS").write_text(f"{meta['sha256']}  tasks.jsonl\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
