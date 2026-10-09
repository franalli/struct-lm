"""Stage 6, step 1: the served model sees the prompt the trainer and the eval rendered, and answers as
the eval's engine did. Runs against a live serve_vllm.sh server (serve/modal_serve.py::bench calls it
after every server start); tests/test_template.py reads its output once pulled.

  python serve/served_check.py --variant bf16 --smoke results/runs/dpo-strict/generations.jsonl \
      --out results/serve/bf16/served_check.jsonl

  template  the 5 SFT template prompts (eval/sft_template_prompts.jsonl, one per format) through
            /v1/chat/completions with return_token_ids: the prompt ids the server built (compared
            with train/sft_data.encode's in the test: one BOS, one [INST], no system prompt) and
            the completion's ids, which must end on </s> (id 2) with finish_reason "stop"
  smoke     (with --smoke) 20 eval items, 5 per task by a fixed hash, with run_eval.generate's
            per-task settings (prompts.GEN + CHAT_GEN, greedy, seed 0): the served text against the
            eval's saved output. Batch shape changes kernels, so identical is not guaranteed; a
            mismatch passes only at a near-tie: the eval's token is in the server's top 2 at the
            first divergence, within NEAR_TIE nats of the server's pick (pre-registered, Stage 6)
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _d in (REPO / "eval", Path("/root/eval")):
    if (_d / "prompts.py").exists():
        sys.path.insert(0, str(_d))
        EVAL = _d
        break
from prompts import CHAT_GEN, GEN
from run_eval import build_items

NEAR_TIE = 0.1  # nats
PER_TASK = 5
TEMPLATE_MAX_TOKENS = 512  # long enough for every template answer to end on </s>


def h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def chat(base_url: str, prompt: str, **params) -> dict:
    body = {
        "model": "struct-lm",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 0,
        "return_token_ids": True,
        **params,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def first_divergence(tokens: list[dict], offline: str) -> dict:
    """Walk the server's tokens along the eval's text. At the first token that doesn't continue it,
    the margin is the server's logprob minus that of the top-2 alternative that does (None if the
    eval's continuation isn't among the top 2: not a near-tie)."""
    pos = 0
    for k, t in enumerate(tokens):
        if offline.startswith(t["token"], pos):
            pos += len(t["token"])
            continue
        alts = [
            a
            for a in t["top_logprobs"]
            if a["token"] != t["token"] and a["token"] and offline.startswith(a["token"], pos)
        ]
        margin = round(t["logprob"] - alts[0]["logprob"], 4) if alts else None
        return {
            "index": k,
            "served": t["token"],
            "offline_next": offline[pos : pos + 24],
            "margin": margin,
        }
    # every served token matched: the served text is a prefix of the eval's (or equal to it)
    return {
        "index": len(tokens),
        "served": None,
        "offline_next": offline[pos : pos + 24],
        "margin": None,
    }


def template_rows(base_url: str) -> list[dict]:
    rows = []
    for line in (EVAL / "sft_template_prompts.jsonl").read_text().splitlines():
        p = json.loads(line)
        r = chat(base_url, p["prompt"], max_tokens=TEMPLATE_MAX_TOKENS)
        c = r["choices"][0]
        rows.append(
            {
                "check": "template",
                "id": p["id"],
                "format": p["format"],
                "prompt_token_ids": r["prompt_token_ids"],
                "token_ids": c["token_ids"],
                "finish_reason": c["finish_reason"],
                "stop_reason": c.get("stop_reason"),
                "text": c["message"]["content"],
            }
        )
    return rows


def smoke_rows(base_url: str, generations: Path) -> list[dict]:
    saved = {(r["task"], r["id"]): r for r in map(json.loads, generations.read_text().splitlines())}
    items = build_items(EVAL / "tasks", None)
    rows = []
    for task in GEN:
        picked = sorted(
            (it for it in items if it["task"] == task), key=lambda it: h(f"smoke:{it['id']}")
        )[:PER_TASK]
        gen = {**GEN[task], **CHAT_GEN.get(task, {})}
        for it in picked:
            want = saved[(task, it["id"])]
            offline = want.get("raw_output", want["output"])  # chat one-line tasks: the full reply
            r = chat(
                base_url,
                it["prompt"],
                max_tokens=gen["max_tokens"],
                stop=gen["stop"] or None,
                logprobs=True,
                top_logprobs=2,
            )
            c = r["choices"][0]
            text = c["message"]["content"]
            row = {
                "check": "smoke",
                "task": task,
                "id": it["id"],
                "identical": text == offline,
                "finish_reason": c["finish_reason"],
                "served": text,
                "offline": offline,
            }
            if text != offline:
                row["divergence"] = first_divergence(c["logprobs"]["content"], offline)
                m = row["divergence"]["margin"]
                row["near_tie"] = m is not None and m < NEAR_TIE
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--smoke", help="the eval's generations.jsonl for this checkpoint (bf16 only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = template_rows(args.base_url)
    if args.smoke:
        rows += smoke_rows(args.base_url, Path(args.smoke))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"variant": args.variant, **r}) + "\n" for r in rows))
    t = [r for r in rows if r["check"] == "template"]
    ends = sum(r["finish_reason"] == "stop" and r["token_ids"][-1:] == [2] for r in t)
    print(f"{args.variant} template: {ends}/{len(t)} end on </s>")
    s = [r for r in rows if r["check"] == "smoke"]
    if s:
        same = sum(r["identical"] for r in s)
        ties = sum(r.get("near_tie", False) for r in s)
        print(
            f"{args.variant} smoke: {same}/{len(s)} identical, {ties} near-ties, "
            f"{len(s) - same - ties} unexplained"
        )


if __name__ == "__main__":
    main()
