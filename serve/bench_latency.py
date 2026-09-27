"""Latency / throughput benchmark against the running vLLM server.

Streams completions and measures, per concurrency level:
  TTFT  time to first token        (prefill-bound; what users feel)
  ITL   inter-token latency        (decode-bound)
  E2E   end-to-end request latency
  tok/s aggregate output throughput
Reports p50/p95. Compare bf16 vs AWQ at the same concurrency for the write-up.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path

from openai import AsyncOpenAI


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    stamps: list[float] = []
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            stamps.append(time.perf_counter())
    end = time.perf_counter()
    gaps = [b - a for a, b in pairwise(stamps)]
    return {
        "ttft": stamps[0] - t0 if stamps else end - t0,
        "itl": statistics.mean(gaps) if gaps else 0.0,
        "e2e": end - t0,
        "n_tokens": len(stamps),
    }


async def run_level(client, model, prompts, concurrency, max_tokens) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(p):
        async with sem:
            return await one_request(client, model, p, max_tokens)

    t0 = time.perf_counter()
    rs = await asyncio.gather(*(bounded(p) for p in prompts))
    wall = time.perf_counter() - t0
    row = {
        "concurrency": concurrency,
        "requests": len(rs),
        "tok_per_s": round(sum(r["n_tokens"] for r in rs) / wall, 1),
    }
    for k in ("ttft", "itl", "e2e"):
        vals = [r[k] for r in rs]
        row[f"{k}_p50_ms"] = round(pct(vals, 0.5) * 1000, 1)
        row[f"{k}_p95_ms"] = round(pct(vals, 0.95) * 1000, 1)
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="struct-lm")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tasks", default="eval/tasks", help="prompts are drawn from KPI tasks")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="awq")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    # The real KPI eval prompts (few-shot QA, vocab, 4-passage grounded), built exactly as
    # run_eval.py builds them, so latency is measured on the prompt lengths the model is judged on.
    # Task rows themselves have no "prompt" field.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from run_eval import build_items

    pool = [it["prompt"] for it in build_items(Path(args.tasks), None)]
    prompts = [pool[i % len(pool)] for i in range(args.requests)]

    await one_request(client, args.model, pool[0], 16)  # warm-up
    rows = [
        await run_level(client, args.model, prompts, c, args.max_tokens) for c in args.concurrency
    ]

    out = Path("results/bench") / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
