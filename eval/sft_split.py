"""Which eval source chunks Stage 3's SFT synthesis may draw on: the "seen" half.

  .venv/bin/python eval/sft_split.py          # -> eval/tasks/sft_seen_chunks.txt

Stage 2 showed that one pooled score can't tell knowledge of the documents a model was taught from
transfer to ones it wasn't (-8% vs -0.4% perplexity). So domain_qa and vocab are scored in two
halves (run_eval.py qa_seen / qa_unseen, vocab_seen / vocab_unseen):
  seen    the item's source chunk is in sft_seen_chunks.txt; SFT synthesis may build examples
          from it (never the eval question itself), so the score measures knowledge injection
  unseen  SFT synthesis must skip the chunk, like every other chunk in eval_chunk_ids.txt, so the
          score measures transfer to facts it was not taught

A chunk is seen when the first byte of sha256("sft-seen:" + chunk_id) is even: about half, and
fixed per chunk, so growing a task never moves an existing item between halves. Never seen:
chunks shown as context in grounded or adversarial items (those tasks test reading, not recall)
and the few-shot items' chunks. Everything else in eval_chunk_ids.txt stays off-limits to SFT.
Before Stage 3 no row has seen anything, so the two halves are a null check.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

TASKS = Path(__file__).parent / "tasks"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (TASKS / f"{name}.jsonl").read_text().splitlines()]


def is_seen(chunk_id: str) -> bool:
    return hashlib.sha256(f"sft-seen:{chunk_id}".encode()).digest()[0] % 2 == 0


def main() -> None:
    reserved = {
        c["chunk_id"] for t in ("grounded", "adversarial") for i in rows(t) for c in i["context"]
    }
    reserved |= {i["source_chunk"] for i in rows("fewshot")}
    halves: Counter = Counter()
    seen = set()
    for task in ("domain_qa", "vocab"):
        for item in rows(task):
            c = item["source_chunk"]
            half = "seen" if c not in reserved and is_seen(c) else "unseen"
            halves[task, half] += 1
            if half == "seen":
                seen.add(c)
    (TASKS / "sft_seen_chunks.txt").write_text("".join(f"{c}\n" for c in sorted(seen)))
    for task in ("domain_qa", "vocab"):
        print(f"{task:9} seen {halves[task, 'seen']:3}  unseen {halves[task, 'unseen']:3}")
    print(f"{len(seen)} chunks -> {TASKS / 'sft_seen_chunks.txt'}")


if __name__ == "__main__":
    main()
