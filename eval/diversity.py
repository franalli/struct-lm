"""Output diversity of a chat checkpoint (Stage 3, B6), and the </s> check on sampled answers (B5).

  .venv/bin/python eval/diversity.py build        # -> eval/diversity_prompts.jsonl (once; git keeps
                                                  #    eval/hosted/, make sft-replay rebuilds it)
  .venv/bin/python eval/diversity.py score <run>  # samples -> results/diversity/<run>.json
  .venv/bin/python eval/diversity.py collapse <run>  # dpo_probe samples -> results/diversity/<run>_collapse.json

The prompts (100): 50 general instructions held out from the Tulu 3 SFT mixture (the replay
source: same eligibility as data/scripts/sft_replay.py, none of the 500 replay rows, none of the
excluded Persona subsets), and 50 domain prompts from the eval (25 grounded, 25 vocab), each set
chosen by a fixed hash. Only the prompts are used; nothing here trains. eval/sample.py's diversity
job samples one answer per prompt at temperature 0.7.

Scored over every answer's tokens (Tekken ids, the same tokenizer for every checkpoint compared):
  distinct_4  unique token 4-grams / all token 4-grams, pooled over the answers; collapse onto
              the teacher's phrasing lowers it
  entropy     Shannon entropy (bits) of the pooled unigram distribution of output tokens
  mean_len    mean output tokens per answer; stop_rate: answers ended by </s>, not the cap
Each also per group (general / domain). B7 reads sft-from-cpt against instruct-8b: within 10% on
distinct_4 and entropy, or the write-up says the teacher's style collapsed the student.
The eos job (20 Stage 4 prompts x 4 samples, temperature 0.8) gives eos_stop_rate; B5 passes at 0.95.
"""

import hashlib
import json
import math
import pathlib
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
PROMPTS = REPO / "eval/diversity_prompts.jsonl"
N_GENERAL, N_GROUNDED, N_VOCAB = 50, 25, 25
EOS_PASS = 0.95


def h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def build() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(REPO / "data/scripts"))
    from run_eval import build_items
    from sft_replay import EXCLUDED, SHARDS, eligible
    from sft_replay import REPO as TULU

    replay = {
        json.loads(line)["id"] for line in (REPO / "data/sft/work/replay.jsonl").open()
    }  # the 500 rows trained on
    best: list[tuple[str, dict]] = []
    for shard in SHARDS:
        path = hf_hub_download(TULU, shard, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=20_000, columns=["id", "source", "messages"]
        ):
            for r in batch.to_pylist():
                if r["id"] in replay or r["source"] in EXCLUDED or not eligible(r):
                    continue
                best.append((h(f"diversity:{r['id']}"), r))
            best = sorted(best, key=lambda x: x[0])[:N_GENERAL]
    rows = [
        {
            "id": r["id"],
            "format": "general",
            "source": r["source"],
            "prompt": r["messages"][0]["content"],
        }
        for _, r in best
    ]
    items = build_items(REPO / "eval/tasks", None, ["grounded", "vocab"])
    for task, n in (("grounded", N_GROUNDED), ("vocab", N_VOCAB)):
        pool = sorted(
            (it for it in items if it["task"] == task), key=lambda it: h(f"diversity:{it['id']}")
        )
        rows += [{"id": it["id"], "format": task, "prompt": it["prompt"]} for it in pool[:n]]
    with PROMPTS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts -> {PROMPTS}: {Counter(r['format'] for r in rows)}")


def metrics(answers: list[list[int]], finish: list[str]) -> dict:
    grams, total, unigrams = set(), 0, Counter()
    for toks in answers:
        unigrams.update(toks)
        for i in range(len(toks) - 3):
            grams.add(tuple(toks[i : i + 4]))
            total += 1
    n = sum(unigrams.values())
    entropy = -sum(c / n * math.log2(c / n) for c in unigrams.values()) if n else 0.0
    return {
        "answers": len(answers),
        "distinct_4": round(len(grams) / total, 4) if total else None,
        "entropy": round(entropy, 4),
        "mean_len": round(n / len(answers), 1) if answers else None,
        "stop_rate": round(sum(f == "stop" for f in finish) / len(finish), 4) if finish else None,
    }


def score(run: str) -> None:
    sdir = REPO / "results/runs" / run / "samples"
    res: dict = {"run": run}
    div = sdir / "diversity.jsonl"
    if div.exists():
        rows = [json.loads(line) for line in div.read_text().splitlines()]
        firsts = [(r["format"], r["samples"][0]) for r in rows]
        res["diversity"] = metrics(
            [s["token_ids"] for _, s in firsts], [s["finish_reason"] for _, s in firsts]
        )
        for group, fmts in (("general", {"general"}), ("domain", {"grounded", "vocab"})):
            sel = [s for f, s in firsts if f in fmts]
            res[f"diversity_{group}"] = metrics(
                [s["token_ids"] for s in sel], [s["finish_reason"] for s in sel]
            )
    eos = sdir / "eos.jsonl"
    if eos.exists():
        flat = [s for line in eos.read_text().splitlines() for s in json.loads(line)["samples"]]
        rate = sum(s["finish_reason"] == "stop" for s in flat) / len(flat)
        res["eos"] = {
            "samples": len(flat),
            "eos_stop_rate": round(rate, 4),
            "mean_len": round(sum(s["n_tokens"] for s in flat) / len(flat), 1),
            "passed": rate >= EOS_PASS,
        }
    out = REPO / "results/diversity" / f"{run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


COLLAPSE_OVERLAP, COLLAPSE_STOP = 0.9, 0.25


def overlap(a: list[int], b: list[int]) -> float:
    """Token overlap of two samples: the multiset intersection of their token ids over the longer
    one's length (1.0 = the same tokens, order aside)."""
    if not a and not b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(len(a), len(b))


def collapse(run: str) -> None:
    """Stage 4's pre-pairing check (stop rule fixed 2026-10-06, before sampling): per prompt of the
    dpo_probe job, whether all 4 samples are near-duplicates (every pairwise token overlap above
    0.9), and the distinct-4 of the 4 samples pooled. Stop if more than a quarter of the
    non-abstain prompts collapse: temperature 1.0 and the system-prompt variants before any pair is
    built. Abstain prompts are out of the share (their correct answer is one fixed sentence, so
    collapse there is the model being right: a data-yield fact, not narrowness) and reported on
    their own line; closed-book stays in the share (a collapse on a wrong answer is the failure
    on-policy DPO can't repair) and is also reported on its own line."""
    from itertools import combinations

    rows = [
        json.loads(line) for line in (REPO / f"results/runs/{run}/samples/dpo_probe.jsonl").open()
    ]
    per = []
    for r in rows:
        toks = [
            s["token_ids"][:-1] if s["finish_reason"] == "stop" else s["token_ids"]
            for s in r["samples"]
        ]
        pairs = [overlap(a, b) for a, b in combinations(toks, 2)]
        grams, total = set(), 0
        for t in toks:
            for i in range(len(t) - 3):
                grams.add(tuple(t[i : i + 4]))
                total += 1
        per.append(
            {
                "id": r["id"],
                "format": r["format"],
                "min_pair_overlap": round(min(pairs), 4),
                "collapsed": all(x > COLLAPSE_OVERLAP for x in pairs),
                "distinct_4": round(len(grams) / total, 4) if total else None,
                "mean_len": round(sum(map(len, toks)) / len(toks), 1),
            }
        )

    def summary(sel: list[dict]) -> dict:
        d4 = [x["distinct_4"] for x in sel if x["distinct_4"] is not None]
        return {
            "prompts": len(sel),
            "collapsed": sum(x["collapsed"] for x in sel),
            "collapsed_share": round(sum(x["collapsed"] for x in sel) / len(sel), 4)
            if sel
            else None,
            "mean_distinct_4": round(sum(d4) / len(d4), 4) if d4 else None,
            "prompts_without_4grams": len(sel) - len(d4),
            "mean_len": round(sum(x["mean_len"] for x in sel) / len(sel), 1) if sel else None,
        }

    gating = [x for x in per if x["format"] != "abstain"]
    res = {
        "run": run,
        "rule": f"collapsed = every pairwise token overlap > {COLLAPSE_OVERLAP}; stop if the "
        f"collapsed share of non-abstain prompts > {COLLAPSE_STOP}",
        "non_abstain": summary(gating),
        "all": summary(per),
        "by_format": {
            f: summary([x for x in per if x["format"] == f])
            for f in sorted({x["format"] for x in per})
        },
        "per_prompt": per,
    }
    res["stop"] = res["non_abstain"]["collapsed_share"] > COLLAPSE_STOP
    out = REPO / "results/diversity" / f"{run}_collapse.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}, indent=2))


def main() -> None:
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["score"] and len(sys.argv) == 3:
        score(sys.argv[2])
    elif sys.argv[1:2] == ["collapse"] and len(sys.argv) == 3:
        collapse(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
