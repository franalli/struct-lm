"""Stage 6 benchmark request sets, built from the KPI eval's own prompts (run_eval.build_items), so
latency is measured on the prompts the model is judged on, in `vllm bench serve`'s custom format.

  .venv/bin/python serve/bench_data.py            # -> serve/bench/*.jsonl + serve/bench_manifest.json
  python serve/bench_data.py --check              # rebuild and compare with the manifest (Modal)

The files are derived, deterministic and gitignored (unique.jsonl is over the repo's 1 MB file
limit); the committed manifest pins each file's sha256, counts and token lengths, and the Modal bench
rebuilds them from eval/tasks/ and refuses to run on a set that doesn't match it.

Rows: {"prompt", "output_tokens", "task", "id"}. `prompt` is the untemplated user text, exactly
what run_eval sends to llm.chat; the server applies the chat template once, so the bench runs with
--skip-chat-template (without it vLLM renders the template client-side and the server renders it
again: [INST] would arrive as text, rule 2's bug). `output_tokens` is the task's eval cap
(prompts.GEN + CHAT_GEN: domain_qa 64, grounded 300, adversarial 200), sent as max_completion_tokens
with --custom-output-len -1; the model stops itself on </s> well before it.

  unique.jsonl           360 requests: 216 domain_qa (closed-book), 108 grounded, 36 adversarial
                         (abstain), 60 / 30 / 10; every prompt distinct. 108 grounded is all the eval
                         has, which sets the size (Stage 6 plan, user decision)
  grounded_unique.jsonl  the 108 grounded prompts as evaluated: 4 passages each, none shared
  grounded_rag.jsonl     the same 108 questions in 27 groups of 4 sharing one context, the group's 4
                         gold passages in a fixed order: the passage block is byte-identical within a
                         group and precedes the question (prompts.grounded_prompt), so a retrieval
                         deployment's repeated context is what the prefix cache can reuse. Each
                         question's own gold passage is in its context, so it stays answerable (no
                         abstain confound); only TTFT is compared against grounded_unique
  closedbook.jsonl       216 domain_qa: the speculative-decoding control (nothing to copy from)

Request order is the file order (the bench runs with --disable-shuffle); grounded_unique and
grounded_rag list the questions in the same order, with group members spread through it.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _d in (REPO / "eval", Path("/root/eval")):
    if (_d / "run_eval.py").exists():
        sys.path.insert(0, str(_d))
        EVAL = _d
        break
from prompts import CHAT_GEN, GEN, grounded_prompt
from run_eval import build_items

MIX = {"domain_qa": 216, "grounded": 108, "adversarial": 36}
GROUP = 4
BASE = "mistralai/Ministral-3-8B-Base-2512"
MANIFEST = Path(__file__).resolve().parent / "bench_manifest.json"


def h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def cap(task: str) -> int:
    return {**GEN[task], **CHAT_GEN.get(task, {})}["max_tokens"]


def row(it: dict, prompt: str | None = None) -> dict:
    return {
        "prompt": prompt or it["prompt"],
        "output_tokens": cap(it["task"]),
        "task": it["task"],
        "id": it["id"],
    }


def pick(items: list[dict], task: str, n: int) -> list[dict]:
    """n items of a task by a fixed hash of their ids (all of them when n is the task's size)."""
    rows = sorted((it for it in items if it["task"] == task), key=lambda it: h(f"bench:{it['id']}"))
    if len(rows) < n:
        raise SystemExit(f"{task}: {len(rows)} items, {n} asked for")
    return rows[:n]


def rag_prompts(grounded: list[dict]) -> dict[str, str]:
    """{id: prompt} with groups of GROUP questions sharing one context: the members' gold passages,
    groups formed in gold-chunk order (neighbouring pages of one document end up together)."""
    by_gold = sorted(grounded, key=lambda it: (it["ref"]["gold_chunk_ids"][0], it["id"]))
    out = {}
    for k in range(0, len(by_gold), GROUP):
        members = by_gold[k : k + GROUP]
        context = []
        for it in members:
            (gold,) = it["ref"]["gold_chunk_ids"]
            context += [c for c in it["ref"]["context"] if c["chunk_id"] == gold]
        for it in members:
            out[it["id"]] = grounded_prompt(it["ref"]["question"], context)
    return out


def build(tasks_dir: Path) -> dict[str, list[dict]]:
    items = build_items(tasks_dir, None, list(MIX))
    chosen = {t: pick(items, t, n) for t, n in MIX.items()}
    unique = [row(it) for t in MIX for it in chosen[t]]
    random.Random(0).shuffle(unique)
    grounded = chosen["grounded"]
    order = sorted(grounded, key=lambda it: h(f"order:{it['id']}"))  # spreads group members
    rag = rag_prompts(grounded)
    closed = [row(it) for it in chosen["domain_qa"]]
    return {
        "unique": unique,
        "grounded_unique": [row(it) for it in order],
        "grounded_rag": [row(it, rag[it["id"]]) for it in order],
        "closedbook": closed,
    }


def dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def chat_lengths(prompts: list[str]) -> list[int]:
    """Prompt length in the ids the server sees: mistral-common's chat rendering of one user turn."""
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_hf_hub(BASE)
    return [
        len(
            tok.encode_chat_completion(
                ChatCompletionRequest(messages=[UserMessage(content=p)])
            ).tokens
        )
        for p in prompts
    ]


def summary(rows: list[dict], lengths: list[int]) -> dict:
    out = {}
    for task in sorted({r["task"] for r in rows}):
        ls = sorted(n for r, n in zip(rows, lengths) if r["task"] == task)
        out[task] = {
            "n": len(ls),
            "prompt_tokens": {
                "min": ls[0],
                "median": statistics.median(ls),
                "p90": ls[int(0.9 * (len(ls) - 1))],
                "max": ls[-1],
            },
            "output_cap": cap(task),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(EVAL / "tasks"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "bench"))
    ap.add_argument(
        "--check", action="store_true", help="refuse files that differ from the manifest"
    )
    args = ap.parse_args()
    sets = build(Path(args.tasks_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, rows in sets.items():
        text = dump(rows)
        (out / f"{name}.jsonl").write_text(text)
        shas[name] = hashlib.sha256(text.encode()).hexdigest()
    if args.check:
        want = {k: v["sha256"] for k, v in json.loads(MANIFEST.read_text())["files"].items()}
        if shas != want:
            raise SystemExit(f"bench sets differ from {MANIFEST.name}: {shas} != {want}")
        print(f"bench sets match {MANIFEST.name} -> {out}")
        return
    files = {}
    for name, rows in sets.items():
        files[name] = {
            "sha256": shas[name],
            "requests": len(rows),
            "tasks": summary(rows, chat_lengths([r["prompt"] for r in rows])),
        }
    MANIFEST.write_text(json.dumps({"tokenizer": BASE, "files": files}, indent=2) + "\n")
    for name, f in files.items():
        print(name, f["requests"], {t: s["prompt_tokens"]["median"] for t, s in f["tasks"].items()})


if __name__ == "__main__":
    main()
