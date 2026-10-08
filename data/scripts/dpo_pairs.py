"""Stage 4 step 6: preference pairs from the scored pool samples -> data/dpo/{train,val}.jsonl,
SHA256SUMS, pairs_meta.json

  .venv/bin/python data/scripts/dpo_pairs.py

Rules fixed 2026-10-08 before sampling, as amended the same evening before any pool sample
existed (notes/decisions.md): DPO on verifiable preferences, no judge in pair-building, no
closed-book cap, no definition pairs.

Per prompt (data/dpo/work/scored.jsonl; dpo_split train -> train.jsonl, val -> val.jsonl):
  chosen    a verifier-correct (closed-book: and form-checked) or rule-passing sample, by a fixed
            hash
  rejected  a verifier-wrong or rule-failing sample, by a fixed hash, whose length keeps the
            chosen at most 1.5x the rejected in tokens (except grounded)
  a second pair from the remaining samples; at most 2 per prompt
prompt + longer completion over 4,096 tokens: the pair is dropped, never truncated. Shortfall:
under 500 pairs, stop and report.

The as-registered count (the closed-book cap at half the pairs, grounded on the registered rules:
cite_valid, every sentence cited, no false abstain) is computed next to the amended one and both
go into pairs_meta.json, so the amendment's effect is visible.

Records: TRL's conversational preference form with an explicit prompt, plus the vLLM token ids
the trainer reads (prompt_ids, chosen_ids, rejected_ids: the exact on-policy tokens). Every
completion is the SFT model's own sample (rule 13: no Claude-written text).
"""

import hashlib
import json
import statistics
import sys
from collections import Counter

from dpo_common import REPO
from sft_common import h01, read_jsonl, write_jsonl

SCORED = REPO / "data/dpo/work/scored.jsonl"
POOL = REPO / "data/dpo/prompts.jsonl"
OUT = REPO / "data/dpo"
MARGIN, FLOOR_PAIRS, SMOL_FLOOR = 2.0, 500, 1000
MAX_TOKENS, LEN_RATIO = 4096, 1.5
PAIRED = ("closed_book", "abstain", "grounded")  # definition: no verifier, no pairs


def eff(s: dict, rules: str) -> float:
    return s["eff_by_rules"][rules] if "eff_by_rules" in s else s["eff"]


def pairs_for(row: dict, rules: str, drops: Counter) -> list[tuple]:
    fmt = row["format"]
    left = [s for s in row["samples"] if s.get("eff") is not None]
    out = []
    while len(out) < 2 and len(left) >= 2:
        cands = [s for s in left if eff(s, rules) == 5.0 and (fmt != "closed_book" or s["form_ok"])]
        if not cands:
            break
        chosen = min(cands, key=lambda s: h01(f"c:{row['id']}:{s['k']}"))
        rejs = [s for s in left if eff(s, rules) <= eff(chosen, rules) - MARGIN
                and s["text"].strip() != chosen["text"].strip()]  # fmt: skip
        if fmt != "grounded":
            fit = [s for s in rejs if chosen["n_tokens"] <= LEN_RATIO * s["n_tokens"]]
            if rejs and not fit:
                drops["length_ratio"] += 1
            rejs = fit
        if not rejs:
            break
        rejected = min(rejs, key=lambda s: h01(f"r:{row['id']}:{s['k']}"))
        if (
            len(row["prompt_token_ids"]) + max(chosen["n_tokens"], rejected["n_tokens"])
            > MAX_TOKENS
        ):
            drops["over_4096"] += 1
            break
        out.append((chosen, rejected, len(out)))
        left = [s for s in left if s is not chosen and s is not rejected]
    return out


def build(rows: list[dict], rules: str, cap: bool) -> tuple[list[tuple], Counter]:
    drops: Counter = Counter()
    made = [(row, c, r, k) for row in rows for c, r, k in pairs_for(row, rules, drops)]
    if cap:  # the registered mix rule: closed-book at most half (second pairs first, then hash)
        cb = [m for m in made if m[0]["format"] == "closed_book"]
        other = len(made) - len(cb)
        if len(cb) > other:
            cb.sort(key=lambda m: (-m[3], h01(f"cap:{m[0]['id']}:{m[3]}")))
            cut = {id(m) for m in cb[: len(cb) - other]}
            drops["closed_book_cap"] += len(cut)
            made = [m for m in made if id(m) not in cut]
    return made, drops


def record(row: dict, pool_row: dict, chosen: dict, rejected: dict, k: int) -> dict:
    prompt = pool_row["prompt"]
    return {
        "id": f"{row['id']}#{k}",
        "prompt_id": row["id"],
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen["text"].strip()}],
        "rejected": [{"role": "assistant", "content": rejected["text"].strip()}],
        "format": row["format"],
        "kind": pool_row["kind"],
        "score_chosen": chosen["eff"],
        "score_rejected": rejected["eff"],
        "label_source": "rule" if row["format"] == "grounded" else "verifier",
        "prompt_hash": hashlib.sha256(prompt[0]["content"].encode()).hexdigest()[:16],
        "system_variant": None,
        "prompt_ids": row["prompt_token_ids"],
        "chosen_ids": chosen["token_ids"],
        "rejected_ids": rejected["token_ids"],
    }


def summary(made: list[tuple], drops: Counter) -> dict:
    return {
        "pairs": len(made),
        "by_format": dict(Counter(m[0]["format"] for m in made)),
        "drops": dict(drops),
    }


def main() -> None:
    pool = {r["id"]: r for r in read_jsonl(POOL)}
    rows = [r for r in read_jsonl(SCORED)
            if pool[r["id"]]["dpo_split"] in ("train", "val") and r["format"] in PAIRED]  # fmt: skip
    registered, reg_drops = build(rows, "registered", cap=True)
    made, drops = build(rows, "amended", cap=False)
    status = "ok" if len(made) >= FLOOR_PAIRS else "stop"

    recs = {"train": [], "val": []}
    for row, chosen, rejected, k in made:
        recs[pool[row["id"]]["dpo_split"]].append(record(row, pool[row["id"]], chosen, rejected, k))
    meta: dict = {
        "status": status,
        "rule": "2026-10-08 amendment: verifier/rule labels only, no closed-book cap, no definitions",
        "amended": summary(made, drops),
        "as_registered": summary(registered, reg_drops),
        "under_smol_floor": len(made) < SMOL_FLOOR,
    }
    for split, rs in recs.items():
        meta[split] = {
            "pairs": len(rs),
            "prompts": len({r["prompt_id"] for r in rs}),
            "by_format": dict(Counter(r["format"] for r in rs)),
            "by_format_source": dict(Counter(f"{r['format']}:{r['label_source']}" for r in rs)),
            "length_ratio_median": {
                f: round(statistics.median(len(r["chosen_ids"]) / len(r["rejected_ids"])
                                       for r in rs if r["format"] == f), 3)
                for f in sorted({r["format"] for r in rs})
            },
        }  # fmt: skip
    for split, rs in recs.items():
        write_jsonl(OUT / f"{split}.jsonl", rs)
    sums = "".join(
        f"{hashlib.sha256((OUT / f'{s}.jsonl').read_bytes()).hexdigest()}  {s}.jsonl\n"
        for s in ("train", "val")
    )
    (OUT / "SHA256SUMS").write_text(sums)
    meta["dataset_hash"] = hashlib.sha256(sums.encode()).hexdigest()
    bench = json.loads((REPO / "results/dpo/judge_bench.json").read_text())
    meta["judge_bench"] = bench["chosen"]
    (OUT / "pairs_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    if status == "stop":
        sys.exit(f"{len(made)} pairs < {FLOOR_PAIRS}: stop and report (2026-10-08 shortfall rule)")


if __name__ == "__main__":
    main()
