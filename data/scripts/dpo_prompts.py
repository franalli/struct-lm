"""Stage 4's prompt pool -> data/dpo/prompts.jsonl (+ prompts_meta.json)

  .venv/bin/python data/scripts/dpo_prompts.py

Two sources, prompts only (no completion is copied: rule 13 concerns the text a model is trained to
produce, and every DPO completion is sampled from the SFT checkpoint):

1. The domain prompts of the Stage 3 SFT train set (data/sft/train.jsonl; sft_val is excluded by
   construction), one row per distinct prompt text, in the set's order. Replay prompts are left out
   (user decision 2026-10-08: no Tulu prompts in DPO).
2. The definitions the Stage 3 cap cut (sft_assemble.DEFINITION_CAP; rejects with rule
   "definition_cap"), never trained on, with their prompts rebuilt exactly as assembly would have
   (sft_assemble.prompt_text). They are appended after the train prompts, in their own
   h01("sft-order:<eid>") order, so the first 20 rows (the pre-registered dpo_probe set) don't move.
   Left out: a term a train definition already defines (term_key), a teacher answer that equals an
   unseen-side eval item from the same document (sft_guard.rule2), and anything the guard blocks.
   Their teacher answers were never read against the passage (7.7% of the 300 that were read were
   defects); they only reach the definition judge as its reference.

Every row gets `dpo_split`, by h01("dpo-split:<id>") within each format: 100 rows `judge` (the
win-rate prompts, shared out by format share, largest remainder), 5% `val` (dpo_val, the loss curve)
and the rest `train`. Both held-out splits are excluded from training.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data/scripts"))

from sft_assemble import prompt_text
from sft_common import h01, read_jsonl
from sft_guard import Guard

SFT = REPO / "data/sft"
OUT = REPO / "data/dpo/prompts.jsonl"
KEEP = ("format", "kind", "origin", "split", "source_chunks", "fact_id")
JUDGE_N = 100
VAL_SHARE = 0.05


def shares(counts: dict[str, int], n: int) -> dict[str, int]:
    """n shared out in proportion to counts, largest remainder (ties by name)."""
    total = sum(counts.values())
    raw = {k: n * v / total for k, v in counts.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def train_prompts() -> tuple[list[dict], Counter]:
    rows, seen, dropped = [], set(), Counter()
    for r in read_jsonl(SFT / "train.jsonl"):
        if r["format"] == "replay":
            dropped["replay"] += 1
            continue
        (msg,) = r["prompt"]
        if msg["content"] in seen:  # paraphrases share facts, not prompt text; exact dupes only
            dropped["duplicate_prompt"] += 1
            continue
        seen.add(msg["content"])
        rows.append(
            {
                "id": r["eid"],
                "prompt": r["prompt"],
                **{k: r[k] for k in KEEP},
                "source": "sft_train",
            }
        )
    return rows, dropped


def cut_definitions(
    guard: Guard, taken: set[str], train_terms: set[str]
) -> tuple[list[dict], Counter, list]:
    cut = {
        r["eid"] for r in read_jsonl(SFT / "sft_rejected.jsonl") if r["rule"] == "definition_cap"
    }
    recs = [e for e in read_jsonl(SFT / "work/judged.jsonl") if e["eid"] in cut]
    assert len(recs) == len(cut), (len(recs), len(cut))
    items = {i.get("id"): i for i in guard.qa + guard.held + guard.vocab}
    rows, dropped, excluded = [], Counter(), []
    for e in sorted(recs, key=lambda e: h01(f"sft-order:{e['eid']}")):
        key = guard.term_key(e["term"])
        hit = guard.rule2(e["doc"], e["answer"], e["term"])
        unseen_hit = hit and items.get(hit, {}).get("source_chunk") not in guard.seen
        text = prompt_text(e, guard)
        if key in train_terms:
            why = "term_in_train"
        elif unseen_hit:
            why = f"rule2_unseen:{hit}"
        elif guard.term_block(e["term"]) or not guard.allowed(e["chunk_id"]):
            why = "guard"
        elif text in taken:
            why = "duplicate_prompt"
        else:
            why = None
        if why:
            dropped[why.split(":")[0]] += 1
            excluded.append({"eid": e["eid"], "term": e["term"], "why": why})
            continue
        taken.add(text)
        train_terms.add(key)  # one definition per term in the pool
        rows.append(
            {
                "id": e["eid"],
                "prompt": [{"role": "user", "content": text}],
                "format": "definition",
                "kind": "term",
                "origin": e["origin"],
                "split": "seen",
                "source_chunks": [e["chunk_id"]],
                "fact_id": e["qid"],
                "source": "definition_cap",
            }
        )
    return rows, dropped, excluded


def assign_splits(rows: list[dict]) -> None:
    by_format = Counter(r["format"] for r in rows)
    judge_n = shares(dict(by_format), JUDGE_N)
    for f, n in by_format.items():
        ranked = sorted(
            (r for r in rows if r["format"] == f), key=lambda r: h01(f"dpo-split:{r['id']}")
        )
        n_val = round(VAL_SHARE * n)
        for k, r in enumerate(ranked):
            r["dpo_split"] = (
                "judge" if k < judge_n[f] else "val" if k < judge_n[f] + n_val else "train"
            )


def main() -> None:
    guard = Guard()
    rows, dropped = train_prompts()
    train_terms = {
        guard.term_key(r["question"])
        for r in read_jsonl(SFT / "train.jsonl")
        if r["format"] == "definition"
    }
    taken = {r["prompt"][0]["content"] for r in rows}
    defs, def_dropped, excluded = cut_definitions(guard, taken, train_terms)
    rows += defs
    assign_splits(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = Counter((r["format"], r["dpo_split"]) for r in rows)
    meta = {
        "sources": {
            "sft_train": "data/sft/train.jsonl",
            "definition_cap": "data/sft/sft_rejected.jsonl",
        },
        "sft_dataset_hash": hashlib.sha256((SFT / "SHA256SUMS").read_bytes()).hexdigest(),
        "prompts": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "dropped_from_sft_train": dict(dropped),
        "dropped_from_definition_cap": dict(def_dropped),
        "excluded_definitions": excluded,
        "by_format": dict(Counter(r["format"] for r in rows).most_common()),
        "by_format_split": {
            f: {s: table[(f, s)] for s in ("train", "val", "judge")}
            for f in dict(Counter(r["format"] for r in rows).most_common())
        },
        "probe_ids": [r["id"] for r in rows[:20]],
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (OUT.parent / "prompts_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "probe_ids"}, indent=2))


if __name__ == "__main__":
    main()
