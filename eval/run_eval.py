"""KPI eval harness: generate with vLLM, score, append a row to results/table.md.

Usage (on a GPU box, or via eval/modal_app.py):
  python eval/run_eval.py --model mistralai/Ministral-3-8B-Base-2512     --run-name base-8b
  python eval/run_eval.py --model mistralai/Ministral-3-8B-Instruct-2512 --run-name instruct-8b --chat
  python eval/run_eval.py --model ... --run-name base-8b --generate-only   # GPU: generations only
  python eval/run_eval.py --run-name base-8b --rescore        # local: all scoring, code + judge, no GPU
  python eval/run_eval.py --model ... --run-name smoke --limit 5 --no-judge   # smoke test

Three modes:
  default          generate, save generations.jsonl, then score (needs a GPU + MISTRAL_API_KEY)
  --generate-only  generate and save, then stop (the Modal half of the split workflow)
  --rescore        skip generation; load generations.jsonl and score it (local half; no GPU)

Layout written under --results-dir:
  runs/<run-name>/generations.jsonl   every prompt and raw output, saved BEFORE scoring
  runs/<run-name>/scored.jsonl        per-item scores and judge reasons
  runs/<run-name>/metrics.json        the aggregate KPIs, lm-eval numbers, model, chat flag
  judge_cache.jsonl                   shared across runs
  table.md                            one row per run

Later stages pass a merged checkpoint path as --model (merge LoRA first; keeps this file
model-agnostic). Regression numbers come from run_lm_eval.sh and are merged in with
--lm-eval-dir.
"""

import argparse
import glob
import json
import pathlib
import statistics
import sys

# Import the sibling modules (prompts, scorers, judge) whether this runs from the repo root, from
# eval/, or as /root/eval/run_eval.py inside the Modal container.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prompts import ABSTAIN_PHRASE, GEN, grounded_prompt, qa_prompt, vocab_prompt
from scorers import abstained, citations_valid, qa_correct

# Task order: files are read, generated and scored in this order.
TASKS = ["domain_qa", "grounded", "vocab", "adversarial"]
# Columns of results/table.md, in order. The first five are the KPIs computed in score():
#   qa_acc          domain_qa: fraction answered correctly (exact or numeric match)
#   cite_valid      grounded: fraction whose citations are all provided ids (rule-based)
#   cite_supported  grounded: fraction the judge finds correct AND supported by cited passages
#   vocab_recall    vocab: fraction of definitions the judge accepts
#   halluc_rate     adversarial: fraction that answered instead of abstaining (lower is better)
# The last three come from lm-eval via merge_lm_eval(), and go blank if it isn't given.
COLUMNS = [
    "qa_acc",
    "cite_valid",
    "cite_supported",
    "vocab_recall",
    "halluc_rate",
    "mmlu",
    "gsm8k",
    "hellaswag",
]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """All rows of a JSONL file, or [] if the file doesn't exist (a task that wasn't generated)."""
    return [json.loads(l) for l in path.open()] if path.exists() else []


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    """Overwrite path with one JSON object per line. ensure_ascii=False keeps symbols like
    φ and ° readable in the committed files."""
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ build prompts ---
def build_items(tasks_dir: pathlib.Path, limit: int | None) -> list[dict]:
    """One work item per task row: {task, id, prompt, ref}.

    `ref` is the untouched task row (gold answer, context, gold ids...), which score() reads.
    `limit` keeps the first N rows of each task, for smoke tests. Prompts are rebuilt from
    eval/tasks/ on every call, including --rescore, which only uses them to line items up
    with saved generations by (task, id)."""
    fewshot = load_jsonl(tasks_dir / "fewshot.jsonl")
    items = []
    for task in TASKS:
        rows = load_jsonl(tasks_dir / f"{task}.jsonl")
        if limit:
            rows = rows[:limit]
        for r in rows:
            if task == "domain_qa":
                prompt = qa_prompt(r["question"], fewshot)
            elif task == "vocab":
                prompt = vocab_prompt(r["term"])
            else:  # grounded, adversarial share the prompt
                prompt = grounded_prompt(r["question"], r["context"])
            items.append({"task": task, "id": r["id"], "prompt": prompt, "ref": r})
    return items


# ---------------------------------------------------------------- generate ---
def generate(
    items: list[dict], model: str, chat: bool, tp: int, max_model_len: int, tokenizer_mode: str
) -> None:
    """Fill it["output"] for every item with vLLM, greedy, one batch per task.

    Batches are per task because each task has its own SamplingParams (max_tokens, stop
    strings; see prompts.GEN). vLLM schedules each batch itself, so there's no batch size.
    vllm is imported here, not at the top, so --rescore runs on a Mac without it."""
    from vllm import LLM, SamplingParams

    # Base models get the raw prompt; chat models get the same text as one user turn, and
    # llm.chat() applies the checkpoint's own template (via mistral-common in "mistral" mode).
    # prompt_final is what gets saved to generations.jsonl. In chat mode it is still the
    # untemplated text: the template is applied inside llm.chat() and never returned.
    for it in items:
        it["prompt_final"] = it["prompt"]

    llm = LLM(
        model=model,
        tokenizer_mode=tokenizer_mode,  # "mistral": Tekken via mistral-common, as trained
        limit_mm_per_prompt={"image": 0},  # Ministral 3 is multimodal; don't reserve vision memory
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=max_model_len,  # 8192 default: grounded prompts carry 4 passages of <=512 tokens
        gpu_memory_utilization=0.9,
        seed=0,
    )
    for task in TASKS:
        batch = [it for it in items if it["task"] == task]
        if not batch:
            continue
        params = SamplingParams(temperature=0.0, seed=0, **GEN[task])
        if chat:
            convs = [[{"role": "user", "content": it["prompt"]}] for it in batch]
            outs = llm.chat(convs, params, use_tqdm=False)
        else:
            outs = llm.generate([it["prompt_final"] for it in batch], params, use_tqdm=False)
        # vLLM returns outputs in input order, so zip lines them back up with their items.
        for it, o in zip(batch, outs):
            it["output"] = o.outputs[0].text
        print(f"generated {len(batch):4d} {task}")


# ------------------------------------------------------------------- score ---
def score(items: list[dict], judge) -> tuple[list[dict], dict]:
    """Score every item; return (per-item records, aggregate metrics).

    `judge` is a judge.Judge or None (--no-judge). Without a judge, the judged metrics
    (cite_supported, vocab_recall) are left out of metrics entirely rather than reported as 0,
    and adversarial items count as not abstained unless they use the exact phrase, so
    halluc_rate is an upper bound in smoke tests.

    If the judge fails on an item after all retries, that item is left out of its judged
    metric (not counted as 0) and marked "judge_failed" in scored.jsonl; metrics["judge_failed"]
    records how many. Nothing is cached for it, so a later --rescore fills it in.

    Each metric is the mean of its bucket: one 0/1 (or bool) entry per scored item."""
    from judge import adversarial_rubric, grounded_rubric, vocab_rubric

    scored, buckets = [], {c: [] for c in COLUMNS}
    n_failed = 0
    for it in items:
        r, out, rec = it["ref"], it["output"], {"task": it["task"], "id": it["id"]}
        if it["task"] == "domain_qa":
            # Rule-based only: exact or numeric match against the gold answer.
            rec["correct"] = qa_correct(
                out, r["answer"], r["answer_type"], r.get("tolerance", 0.02)
            )
            buckets["qa_acc"].append(rec["correct"])
        elif it["task"] == "grounded":
            # Two scores: citation format (rules), then correctness and support (judge).
            ids = {c["chunk_id"] for c in r["context"]}
            rec["cite_valid"] = citations_valid(out, ids)
            buckets["cite_valid"].append(rec["cite_valid"])
            if judge:
                # Cached on the full rubric prompt: the same answer to the same item gets the
                # same verdict across runs.
                v = judge(
                    "grounded",
                    grounded_rubric(r["question"], r["context"], r["gold_chunk_ids"], out),
                )
                if v is None:
                    rec["judge_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["supported"], rec["judge_reason"] = v["score"], v["reason"]
                    buckets["cite_supported"].append(v["score"])
        elif it["task"] == "vocab":
            # Judge-only: definitions vary too much in wording for string matching.
            if judge:
                v = judge("vocab", vocab_rubric(r["term"], r["definition"], out))
                if v is None:
                    rec["judge_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["correct"], rec["judge_reason"] = v["score"], v["reason"]
                    buckets["vocab_recall"].append(v["score"])
        elif it["task"] == "adversarial":
            # The exact abstain phrase is scored without a judge call; any other wording goes
            # to the judge, which also accepts paraphrased refusals.
            if abstained(out, ABSTAIN_PHRASE):
                rec["abstained"], rec["judge_reason"] = 1, "exact abstain phrase"
            elif judge:
                v = judge("adversarial", adversarial_rubric(r["question"], r["context"], out))
                if v is None:
                    rec["judge_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["abstained"], rec["judge_reason"] = v["score"], v["reason"]
            else:
                rec["abstained"] = 0
            if "abstained" in rec:  # absent only when the judge failed on this item
                buckets["halluc_rate"].append(1 - rec["abstained"])
        rec["output"] = out
        scored.append(rec)

    # Empty buckets (lm-eval columns, judged metrics under --no-judge) are left out, so
    # append_table() writes a blank cell instead of a misleading 0.
    metrics: dict = {c: round(statistics.fmean(v), 4) for c, v in buckets.items() if v}
    metrics["n"] = {t: sum(1 for it in items if it["task"] == t) for t in TASKS}
    if n_failed:
        metrics["judge_failed"] = n_failed
    return scored, metrics


def merge_lm_eval(metrics: dict, lm_eval_dir: str | None, run_name: str) -> None:
    """Pull mmlu / gsm8k / hellaswag out of an lm-evaluation-harness results json, if present.

    lm-eval writes <dir>/<run-name>/<model-slug>/results_<timestamp>.json. If a run was
    repeated there are several; the timestamps sort chronologically, so files[-1] is the newest.
    Metric keys are "<metric>,<filter>"; each pick() lists preferred keys first."""
    if not lm_eval_dir:
        return
    files = sorted(glob.glob(f"{lm_eval_dir}/{run_name}/**/results*.json", recursive=True))
    if not files:
        print(f"no lm-eval results under {lm_eval_dir}/{run_name}")
        return
    res = json.loads(pathlib.Path(files[-1]).read_text())["results"]

    def pick(task: str, *keys: str) -> float | None:
        r = res.get(task, {})
        for k in keys:
            if k in r:
                return round(float(r[k]), 4)
        return None

    metrics["mmlu"] = pick("mmlu", "acc,none")  # the group average over the 57 subjects
    # strict-match needs the "#### <answer>" format; flexible-extract (last number) is the fallback
    metrics["gsm8k"] = pick("gsm8k", "exact_match,strict-match", "exact_match,flexible-extract")
    # acc_norm (length-normalised log-likelihood) is the standard HellaSwag number
    metrics["hellaswag"] = pick("hellaswag", "acc_norm,none", "acc,none")


def append_table(table: pathlib.Path, run_name: str, metrics: dict) -> None:
    """Append one row to results/table.md, writing the header first if the file is new.
    Append-only: re-scoring a run adds a second row, so delete the stale one by hand."""
    header = "| run | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 1) + "\n"
    if not table.exists():
        table.write_text(header)
    cells = [f"{metrics[c]:.3f}" if metrics.get(c) is not None else "" for c in COLUMNS]
    with table.open("a") as f:
        f.write(f"| {run_name} | " + " | ".join(cells) + " |\n")


# -------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF id or local path (merged checkpoint for later stages)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tasks-dir", default="eval/tasks")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--chat", action="store_true", help="wrap prompts in the model's chat template")
    ap.add_argument("--rescore", action="store_true", help="re-score saved generations, no GPU")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judging (smoke tests)")
    ap.add_argument(
        "--generate-only",
        action="store_true",
        help="write generations.jsonl and stop; score later with --rescore",
    )
    ap.add_argument("--limit", type=int, help="items per task, for smoke tests")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument(
        "--tokenizer-mode",
        default="mistral",
        help="vLLM tokenizer mode; 'auto' for non-Mistral-3 checkpoints",
    )
    ap.add_argument(
        "--lm-eval-dir", help="e.g. results/lm_eval; merges <dir>/<run-name>/**/results*.json"
    )
    args = ap.parse_args()

    results = pathlib.Path(args.results_dir)
    run_dir = results / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"

    if args.rescore:
        # Re-attach saved outputs to freshly built items by (task, id). Items with no saved
        # generation are dropped, so --rescore with --limit scores that subset only.
        saved = {(r["task"], r["id"]): r["output"] for r in load_jsonl(gen_path)}
        items = build_items(pathlib.Path(args.tasks_dir), args.limit)
        items = [it for it in items if (it["task"], it["id"]) in saved]
        for it in items:
            it["output"] = saved[(it["task"], it["id"])]
    else:
        if not args.model:
            ap.error("--model is required unless --rescore")
        items = build_items(pathlib.Path(args.tasks_dir), args.limit)
        generate(items, args.model, args.chat, args.tp, args.max_model_len, args.tokenizer_mode)
        # Save before scoring: GPU time is the expensive part, so a judge or scoring failure
        # never costs a regeneration (fix it and --rescore).
        write_jsonl(
            gen_path,
            [
                {
                    "task": it["task"],
                    "id": it["id"],
                    "prompt": it["prompt_final"],
                    "output": it["output"],
                }
                for it in items
            ],
        )
        print(f"saved {len(items)} generations -> {gen_path}")
        if args.generate_only:
            return

    # The judge is created only when needed, so --no-judge runs need no API key or client.
    judge = None
    if not args.no_judge:
        from judge import Judge

        judge = Judge(results / "judge_cache.jsonl")

    scored, metrics = score(items, judge)
    merge_lm_eval(metrics, args.lm_eval_dir, args.run_name)
    # On --rescore, keep the model/chat recorded by the generating run unless given again.
    # A --generate-only run writes no metrics.json, so pass --model/--chat when scoring it.
    prev = (
        json.loads((run_dir / "metrics.json").read_text())
        if args.rescore and (run_dir / "metrics.json").exists()
        else {}
    )
    metrics["model"] = args.model or prev.get("model")
    metrics["chat"] = args.chat or prev.get("chat", False)
    write_jsonl(run_dir / "scored.jsonl", scored)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    append_table(results / "table.md", args.run_name, metrics)
    if judge:
        print(f"judge calls this run: {judge.calls}")  # cache misses only; hits are free
        if judge.failures:
            print(
                f"WARNING: judge failed on {judge.failures} items (last error: "
                f"{judge.last_error}). They are left out of the judged metrics and not cached; "
                f"fix the cause and --rescore to fill them in."
            )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
