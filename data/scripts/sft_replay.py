"""General-instruction replay for SFT -> data/sft/work/replay.jsonl

  set -a; . ./.env; set +a
  .venv/bin/python data/scripts/sft_replay.py          # downloads the 6 parquet shards (1.4 GB, HF cache)

500 examples from the Tulu 3 SFT mixture (allenai/tulu-3-sft-mixture, ODC-BY; 939,343 rows), kept
as they are: the general instruction following the domain set lacks, the chat counterpart of the
FineWeb-Edu replay in CPT. Eligible: one user turn and one assistant turn (no system message),
at most 6,000 characters (~1,500 tokens), mostly ASCII text (>= 95%), and not the multilingual
aya subset. Stratified by `source`, proportional to each source's eligible rows (largest
remainder), sampled with a fixed seed. The rows are sorted by source in the shards, so the sample
needs every shard: pass 1 counts eligible rows, pass 2 reads the chosen ones.

No training record may hold Claude-written text, so three Persona subsets are left out (EXCLUDED).
The first draw is kept as it was; its picks from those subsets are dropped and refilled from the
other sources, proportional to their eligible rows, with a second seed. The kept picks don't move,
and the result is still proportional over the allowed sources.
"""

import random
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sft_common import WORK, quotas, update_stats, write_jsonl

REPO = "allenai/tulu-3-sft-mixture"
SHARDS = [f"data/train-{k:05d}-of-00006.parquet" for k in range(6)]
N = 500
MAX_CHARS = 6000
# Responses written by Claude, or by a model the source doesn't name, in the pipeline that used it:
# Persona Python's solutions are claude-3-5-sonnet's (Tulu 3 report, sec. "persona-driven"); the
# Persona MATH card says "Outputs were generated using GPT-4o and Claude 3.5 Sonnet" (no per-row
# model); Persona IF names no response generator. Persona GSM / Algebra solutions are GPT-4o's.
EXCLUDED = {
    "ai2-adapt-dev/personahub_code_v2_34999",
    "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
}


def eligible(r: dict) -> bool:
    m = r["messages"]
    if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant" or "aya" in r["source"]:
        return False
    text = m[0]["content"] + m[1]["content"]
    return 0 < len(text) <= MAX_CHARS and sum(ord(ch) < 128 for ch in text) / len(text) >= 0.95


def main() -> None:
    paths = [hf_hub_download(REPO, s, repo_type="dataset") for s in SHARDS]
    where = defaultdict(list)  # source -> [(shard, row)]
    total = Counter()
    for k, path in enumerate(paths):
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=20_000, columns=["source", "messages"]
        ):
            for r in batch.to_pylist():
                total[r["source"]] += 1
                if eligible(r):
                    where[r["source"]].append((k, row))
                row += 1
        print(f"shard {k}: {row:,} rows")
    quota = quotas({s: len(v) for s, v in where.items()}, N)
    rng = random.Random("sft-replay")
    picks = [(s, k, row) for s in sorted(quota) for k, row in rng.sample(where[s], quota[s])]
    kept = [p for p in picks if p[0] not in EXCLUDED]
    refill = quotas({s: len(v) for s, v in where.items() if s not in EXCLUDED}, N - len(kept))
    taken = {(k, row) for _, k, row in kept}
    rng = random.Random("sft-replay-refill")
    for s in sorted(refill):
        pool = [x for x in where[s] if x not in taken]
        kept += [(s, k, row) for k, row in rng.sample(pool, refill[s])]
    print("refill:", {s: q for s, q in refill.items() if q})
    chosen = defaultdict(list)  # shard -> rows
    for _, k, row in kept:
        chosen[k].append(row)
    out = []
    for k, rows in sorted(chosen.items()):
        t = pq.read_table(paths[k], columns=["id", "source", "messages"]).take(sorted(rows))
        out += t.to_pylist()
    out.sort(key=lambda r: r["id"])
    assert len(out) == N and not {r["source"] for r in out} & EXCLUDED
    write_jsonl(WORK / "replay.jsonl", out)
    stats = {
        "dataset": REPO,
        "license": "ODC-BY-1.0",
        "rows": sum(total.values()),
        "eligible": sum(len(v) for v in where.values()),
        "excluded_sources": {s: len(where[s]) for s in sorted(EXCLUDED)},
        "refilled": N - len(picks) + sum(p[0] in EXCLUDED for p in picks),
        "sampled": len(out),
        "by_source": dict(Counter(r["source"] for r in out).most_common()),
    }
    update_stats("replay", stats)
    print(stats)


if __name__ == "__main__":
    main()
