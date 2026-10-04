"""Closed-book domain_qa through the Mistral API: a frontier reference row for qa_acc.

  set -a; . ./.env; set +a
  .venv/bin/python eval/api_eval.py --model mistral-large-2512 --run-name mistral-large-3
  .venv/bin/python eval/run_eval.py --run-name mistral-large-3 --rescore --chat --allow-partial \
      --model mistral-large-2512

The same 325 closed-book questions and prompt (prompts.qa_prompt) as every other row, sent as one
user turn the way run_eval.py serves chat models, greedy, 64 tokens, and the first line with
content kept (scorers.answer_line). It shows what strong general knowledge reaches on these
questions without the documents. Only domain_qa is generated, so the row's other task columns
stay blank, and gold_lp needs prompt log-probabilities the API doesn't return. Mistral Large 3 also
wrote and verified these questions from the passages; each call is stateless, so that gives it no
answers, though a model's phrasing may suit its own knowledge.

Paced at the key's 30 requests a minute and resumable: answered items are kept in
generations.partial.jsonl until all are done.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prompts import CHAT_GEN
from run_eval import build_items, write_jsonl
from scorers import answer_line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-large-2512")
    ap.add_argument("--run-name", default="mistral-large-3")
    ap.add_argument("--tasks-dir", default="eval/tasks")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--rpm", type=int, default=30)
    args = ap.parse_args()
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=120_000)
    run_dir = pathlib.Path(args.results_dir) / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    partial = run_dir / "generations.partial.jsonl"
    done = {}
    if partial.exists():
        done = {r["id"]: r for r in map(json.loads, partial.read_text().splitlines())}
    items = build_items(pathlib.Path(args.tasks_dir), None, ["domain_qa"])
    interval = 60 / args.rpm * 1.03
    with partial.open("a") as f:
        for n, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            for attempt in range(6):
                start = time.monotonic()
                try:
                    r = client.chat.complete(
                        model=args.model,
                        messages=[{"role": "user", "content": it["prompt"]}],
                        temperature=0.0,
                        max_tokens=CHAT_GEN["domain_qa"]["max_tokens"],
                    )
                    msg = r.choices[0].message
                    content = msg.content if msg else ""
                    raw = (
                        content
                        if isinstance(content, str)
                        else "".join(getattr(c, "text", "") for c in content or [])
                    )
                    break
                except Exception as e:  # noqa: BLE001  rate limits and network errors are retried
                    print(f"  {it['id']} retry {attempt + 1}: {str(e)[:120]}")
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit(f"{it['id']}: no answer after 6 attempts; rerun to resume")
            row = {
                "task": "domain_qa",
                "id": it["id"],
                "prompt": it["prompt"],
                "output": answer_line(raw),
                "raw_output": raw,
            }
            done[it["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(items)}")
            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    rows = [done[it["id"]] for it in items]
    write_jsonl(run_dir / "generations.jsonl", rows)
    meta = {
        "domain_qa": {
            "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "model": f"{args.model} (Mistral API)",
            "chat": True,
            "n": len(rows),
            "prompts_sha256": hashlib.sha256(
                "\n".join(r["prompt"] for r in rows).encode()
            ).hexdigest(),
        }
    }
    (run_dir / "generations_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    partial.unlink()
    print(f"saved {len(rows)} generations -> {run_dir / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
