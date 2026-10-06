"""Stage 4's prompt pool -> data/dpo/prompts.jsonl

  .venv/bin/python data/scripts/dpo_prompts.py

The prompts of the Stage 3 SFT train set (data/sft/train.jsonl; sft_val is excluded by
construction), one row per distinct prompt text, with the record's format, kind, origin, split,
source chunks and eid. Stage 4 samples completions from the SFT checkpoint on these, so it starts
with sampling, not data work. Prompts only: no completion is copied (rule 13 concerns the text a
model is trained to produce). The SFT set's hash is recorded in the file's sidecar so the pool can be
traced to the frozen set it came from.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SFT = REPO / "data/sft"
OUT = REPO / "data/dpo/prompts.jsonl"
KEEP = ("format", "kind", "origin", "split", "source_chunks", "fact_id", "replay_source")


def main() -> None:
    rows, seen = [], set()
    for line in (SFT / "train.jsonl").read_text().splitlines():
        r = json.loads(line)
        (msg,) = r["prompt"]
        if msg["content"] in seen:  # paraphrases share facts, not prompt text; exact dupes only
            continue
        seen.add(msg["content"])
        rows.append({"id": r["eid"], "prompt": r["prompt"], **{k: r[k] for k in KEEP}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "source": "data/sft/train.jsonl",
        "sft_dataset_hash": hashlib.sha256((SFT / "SHA256SUMS").read_bytes()).hexdigest(),
        "prompts": len(rows),
        "duplicates_dropped": sum(1 for _ in (SFT / "train.jsonl").open()) - len(rows),
        "by_format": dict(Counter(r["format"] for r in rows).most_common()),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (OUT.parent / "prompts_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
