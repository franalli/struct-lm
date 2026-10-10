"""The README's demo client: one grounded question with its passages, answered with a [chunk_id]
citation, and one question the passages can't answer, declined with the abstain sentence.

  export DEMO_URL=https://<workspace>--struct-lm-demo-serve.modal.run DEMO_API_KEY=...  # off screen
  .venv/bin/python serve/demo.py                    # both items
  .venv/bin/python serve/demo.py --items gr-0038    # the main clip only

The URL and key come from the environment, so nothing secret is typed on screen. The screen is
cleared at the start and again after a hold at the end, so a recording starts and ends on the same
(blank) frame and loops.

The prompts are the KPI eval's own (eval/prompts.grounded_prompt, one user turn, no system prompt),
for two eval items dpo-strict got right in its saved generations (gr-0038, cited and judged
supported; adv-0050, the exact abstain sentence). They illustrate the two behaviours SFT taught;
the rates are in the README's tables. Greedy, streamed, so the answer appears as it is decoded.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from prompts import grounded_prompt

ITEMS = [("grounded", "gr-0038"), ("adversarial", "adv-0050")]


def item(task: str, item_id: str) -> dict:
    for line in (REPO / "eval/tasks" / f"{task}.jsonl").open():
        r = json.loads(line)
        if r["id"] == item_id:
            return r
    raise SystemExit(f"{item_id} not in eval/tasks/{task}.jsonl")


def ask(url: str, key: str, prompt: str) -> None:
    body = {
        "model": "struct-lm",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {key}"}
    with requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, stream=True,
                       timeout=300) as r:  # fmt: skip
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.decode()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            print(delta, end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=[i for _, i in ITEMS], help="item ids, in order")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between the questions")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds the last answer stays up")
    args = ap.parse_args()
    key = os.environ.get("DEMO_API_KEY") or sys.exit("set DEMO_API_KEY")
    url = (os.environ.get("DEMO_URL") or sys.exit("set DEMO_URL")).rstrip("/")
    task_of = {i: t for t, i in ITEMS}
    chosen = [(task_of[i], i) for i in args.items if i in task_of] or sys.exit("unknown --items")
    print("\033[2J\033[H", end="", flush=True)  # a blank first frame
    for n, (task, item_id) in enumerate(chosen):
        it = item(task, item_id)
        print(f"\n\033[1mQuestion:\033[0m {it['question']}")
        for c in it["context"]:
            print(f"  \033[2m[{c['chunk_id']}] {c['text'][:90]}…\033[0m")
        print("\033[1mAnswer:\033[0m ", end="", flush=True)
        ask(url, key, grounded_prompt(it["question"], it["context"]))
        time.sleep(args.pause if n < len(chosen) - 1 else args.hold)
    print("\033[2J\033[H", end="", flush=True)  # and the same blank last frame


if __name__ == "__main__":
    main()
