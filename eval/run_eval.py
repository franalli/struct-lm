"""KPI eval harness: generate with vLLM, score, append a row to results/table.md.

Usage (on a GPU box, or via eval/modal_app.py):
  python eval/run_eval.py --model mistralai/Ministral-3-8B-Base-2512     --run-name base-8b
  python eval/run_eval.py --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --run-name instruct-8b --chat
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
import datetime
import glob
import hashlib
import json
import pathlib
import re
import statistics
import sys

# Import the sibling modules (prompts, scorers, judge) whether this runs from the repo root, from
# eval/, or as /root/eval/run_eval.py inside the Modal container.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prompts import ABSTAIN_PHRASE, CHAT_GEN, GEN, grounded_prompt, qa_prompt, vocab_prompt
from qa_rules import answer_kind
from scorers import abstained, answer_line, citations, citations_valid, qa_correct, substance

# Task order: files are read, generated and scored in this order.
TASKS = ["domain_qa", "grounded", "vocab", "adversarial"]
# Columns of results/table.md, in order. The first twelve are the KPIs computed in score():
#   qa_acc          domain_qa: fraction answered correctly (exact or numeric match)
#   qa_num, qa_ident, qa_term  the same, by answer kind (qa_rules.answer_kind, or the item's
#                     answer_kind field from make_tasks --task-version 2): values, identifiers
#                     (document ids, article numbers: arbitrary strings, the slowest to learn),
#                     and terms plus everything else
#   qa_seen, qa_unseen  the same, split by whether the item's source chunk is in
#                     eval/tasks/sft_seen_chunks.txt, the half SFT synthesis may draw on
#                     (eval/sft_split.py). Seen measures knowledge injection, unseen transfer;
#                     before Stage 3 nothing is seen, so the halves are a null check
#   grounded_acc    grounded: fraction the judge finds correct, citations ignored
#   cite_valid      grounded: fraction whose citations are all provided ids (rule-based)
#   cite_supported  grounded: fraction the judge finds correct AND supported by cited passages
#   vocab_recall    vocab: fraction of definitions the judge accepts
#   vocab_seen, vocab_unseen  vocab_recall split the same way as qa_seen / qa_unseen
#   halluc_rate     adversarial: fraction that answered instead of abstaining (lower is better)
#   false_abstain   grounded: fraction answered with the abstain phrase although the passages hold
#                     the answer (rule-based, scorers.abstained; lower is better). From Stage 3,
#                     whose SFT set teaches the phrase: the cost side of a lower halluc_rate
#   gold_lp, gold_lp_seen, gold_lp_unseen  domain_qa: mean log-probability, in nats per item, of
#                     the gold answer (eval/gold_lp.py; higher is better), all items and by half;
#                     computed in the generation engine, so blank for runs generated earlier
# The next seven come from lm-eval via merge_lm_eval() (MMLU overall, then its four groups), and
# the last four from eval/perplexity.py via merge_ppl() (lower is better; ppl_postcutoff is the
# 13 federal reports published after the base model, eval/exposure_sources.csv); either set goes
# blank if its directory isn't given or has nothing for the run.
COLUMNS = [
    "qa_acc",
    "qa_num",
    "qa_ident",
    "qa_term",
    "qa_seen",
    "qa_unseen",
    "grounded_acc",
    "cite_valid",
    "cite_supported",
    "vocab_recall",
    "vocab_seen",
    "vocab_unseen",
    "halluc_rate",
    "false_abstain",
    "gold_lp",
    "gold_lp_seen",
    "gold_lp_unseen",
    "mmlu",
    "mmlu_stem",
    "mmlu_hum",
    "mmlu_soc",
    "mmlu_other",
    "gsm8k",
    "hellaswag",
    "ppl_train",
    "ppl_domain_val",
    "ppl_general_val",
    "ppl_postcutoff",
]
PPL = ("ppl_train", "ppl_domain_val", "ppl_general_val", "ppl_postcutoff")
# Checkpoints trained on chat-format data (Stage 3 on: run names and merged paths like sft-from-cpt,
# smoke-sft, dpo-..., grpo-...), which the KPI eval must run with --chat (rule 2), like Instruct.
CHAT_STAGE = re.compile(r"(^|[/_-])(sft|dpo|grpo)([/_-]|$)")


def needs_chat(*names: str | None) -> bool:
    """True when a model id / path or run name is a chat checkpoint: Instruct or Stage 3+."""
    return any(n and ("instruct" in n.lower() or CHAT_STAGE.search(n.lower())) for n in names)


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
def build_items(tasks_dir: pathlib.Path, limit: int | None, tasks: list[str] = TASKS) -> list[dict]:
    """One work item per task row: {task, id, prompt, ref}.

    `ref` is the untouched task row (gold answer, context, gold ids...), which score() reads.
    `limit` keeps the first N rows of each task, for smoke tests. Prompts are rebuilt from
    eval/tasks/ on every call, including --rescore, which only uses them to line items up
    with saved generations by (task, id)."""
    fewshot = load_jsonl(tasks_dir / "fewshot.jsonl")
    items = []
    for task in tasks:
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
    items: list[dict],
    model: str,
    chat: bool,
    tp: int,
    max_model_len: int,
    tokenizer_mode: str,
    config_format: str = "hf",
    kv_cache_dtype: str = "auto",
) -> None:
    """Fill it["output"] for every item with vLLM, greedy, one batch per task.

    Batches are per task because each task has its own SamplingParams (max_tokens, stop
    strings; see prompts.GEN). vLLM schedules each batch itself, so there's no batch size.
    vllm is imported here, not at the top, so --rescore runs on a Mac without it."""
    from vllm import SamplingParams

    # Base models get the raw prompt; chat models get the same text as one user turn, and
    # llm.chat() applies the checkpoint's own template (via mistral-common in "mistral" mode).
    # prompt_final is what gets saved to generations.jsonl. In chat mode it is still the
    # untemplated text: the template is applied inside llm.chat() and never returned.
    for it in items:
        it["prompt_final"] = it["prompt"]

    llm = make_llm(model, tp, max_model_len, tokenizer_mode, config_format, kv_cache_dtype)
    for task in TASKS:
        batch = [it for it in items if it["task"] == task]
        if not batch:
            continue
        gen = {**GEN[task], **CHAT_GEN.get(task, {})} if chat else GEN[task]
        params = SamplingParams(temperature=0.0, seed=0, **gen)
        if chat:
            convs = [[{"role": "user", "content": it["prompt"]}] for it in batch]
            outs = llm.chat(convs, params, use_tqdm=False)
        else:
            outs = llm.generate([it["prompt_final"] for it in batch], params, use_tqdm=False)
        # vLLM returns outputs in input order, so zip lines them back up with their items.
        for it, o in zip(batch, outs):
            it["output"] = o.outputs[0].text
            # "stop" (EOS or a stop string) or "length" (hit max_tokens): whether the model ends
            # its answers, which Stage 2's CPT weakened (docs/stage6.md)
            it["finish_reason"] = o.outputs[0].finish_reason
            if chat and task in CHAT_GEN:  # no "\n" stop was used: keep the answer line only
                it["raw_output"], it["output"] = it["output"], answer_line(it["output"])
        print(f"generated {len(batch):4d} {task}")

    gold_logprobs(llm, [it for it in items if it["task"] == "domain_qa"], model, chat)


GOLD_FIELDS = ("gold_lp", "gold_tokens", "gold_lp_tokens", "gold_lp_end")


def make_llm(
    model: str,
    tp: int,
    max_model_len: int,
    tokenizer_mode: str,
    config_format: str,
    kv_cache_dtype: str = "auto",
    **engine,
):
    """engine: extra vLLM engine arguments (eval/sample.py's pool-sized jobs only).
    kv_cache_dtype: "fp8" for Stage 6's FP8-KV serving variant, which changes attention numerics
    and so is quality-gated like a weight format; "auto" (the model's dtype) everywhere else.
    Quantized checkpoints (compressed-tensors) need no flag: vLLM reads their config."""
    from vllm import LLM

    return LLM(
        model=model,
        tokenizer_mode=tokenizer_mode,  # "mistral": Tekken via mistral-common, as trained
        limit_mm_per_prompt={"image": 0},  # Ministral 3 is multimodal; don't reserve vision memory
        # HF-format weights and config for every checkpoint: a hub repo that also ships Mistral's
        # native params.json + consolidated.safetensors is otherwise loaded through vLLM's native
        # implementation (PixtralForConditionalGeneration), a merged checkpoint through the HF one
        # (Mistral3ForConditionalGeneration), and their deltas would mix training with the path.
        # "auto" (the hub's native path) only to extend a run that generated that way: Stage 0's
        # base-8b and instruct-8b; config_format=hf can't load a hub repo's consolidated weights.
        config_format=config_format,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=max_model_len,  # 8192 default: grounded prompts carry 4 passages of <=512 tokens
        gpu_memory_utilization=0.9,
        seed=0,
        kv_cache_dtype=kv_cache_dtype,
        **engine,
    )


def gold_logprobs(llm, qa: list[dict], model: str, chat: bool) -> None:
    """Gold-answer log-probability for domain_qa items (eval/gold_lp.py), in an engine already
    loaded: prompt + gold answer as one sequence, the answer tokens' prompt logprobs. Sets
    gold_lp (their sum, the end marker included), gold_tokens, gold_lp_tokens (per token) and
    gold_lp_end (the end marker's: "\\n\\n" for base, EOS for chat), so the answer alone is
    gold_lp - gold_lp_end."""
    if not qa:
        return
    from gold_lp import gold_token_ids
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    encoded, failed = gold_token_ids(model, chat, qa)
    if failed:
        print(f"gold_lp: prompt not a token prefix of prompt + answer, left out: {failed}")
    ids = [it["id"] for it in qa if it["id"] in encoded]
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=encoded[i][0]) for i in ids],
        SamplingParams(max_tokens=1, prompt_logprobs=0),
        use_tqdm=False,
    )
    by_id = {it["id"]: it for it in qa}
    for i, o in zip(ids, outs):
        full, n = encoded[i]
        lps = [round(o.prompt_logprobs[k][full[k]].logprob, 4) for k in range(n, len(full))]
        it = by_id[i]
        it["gold_lp"], it["gold_tokens"] = round(sum(lps), 4), len(lps)
        it["gold_lp_tokens"], it["gold_lp_end"] = lps, lps[-1]
    print(f"gold_lp {len(ids):4d} domain_qa ({len(failed)} left out)")


# ------------------------------------------------------------------- score ---
def score(items: list[dict], judge, seen: set[str] | None = None) -> tuple[list[dict], dict]:
    """Score every item; return (per-item records, aggregate metrics).

    `seen` is the set of source chunks SFT synthesis may use (sft_seen_chunks.txt); with it,
    domain_qa and vocab are also scored in seen / unseen halves. None leaves those columns blank.

    `judge` is a judge.Judge or None (--no-judge). Without a judge, the judged metrics
    (grounded_acc, cite_supported, vocab_recall) are left out of metrics entirely rather than
    reported as 0, and adversarial items count as not abstained unless they use the exact
    phrase, so halluc_rate is an upper bound in smoke tests.

    If the judge fails on an item after all retries, that item is left out of its judged
    metric (not counted as 0) and marked "judge_failed" in scored.jsonl; metrics["judge_failed"]
    records how many. Nothing is cached for it, so a later --rescore fills it in.

    Each metric is the mean of its bucket: one 0/1 (or bool) entry per scored item."""
    from judge import adversarial_rubric, grounded_acc_rubric, grounded_rubric, vocab_rubric

    scored, buckets = [], {c: [] for c in COLUMNS}
    n_failed = 0

    def split(prefix: str, ref: dict, value) -> None:
        """Also count a domain_qa / vocab score in its seen or unseen half."""
        if seen is not None:
            buckets[f"{prefix}_{'seen' if ref['source_chunk'] in seen else 'unseen'}"].append(value)

    for it in items:
        r, out, rec = it["ref"], it["output"], {"task": it["task"], "id": it["id"]}
        if it["task"] == "domain_qa":
            # Rule-based only: exact or numeric match against the gold answer.
            rec["correct"] = qa_correct(
                out, r["answer"], r["answer_type"], r.get("tolerance", 0.02)
            )
            buckets["qa_acc"].append(rec["correct"])
            kind = r.get("answer_kind") or answer_kind(r["answer"], r["answer_type"])
            col = {"number": "qa_num", "identifier": "qa_ident"}.get(kind, "qa_term")
            buckets[col].append(rec["correct"])
            split("qa", r, rec["correct"])
            if it.get("gold_lp") is not None:
                rec["gold_lp"] = it["gold_lp"]
                buckets["gold_lp"].append(it["gold_lp"])
                split("gold_lp", r, it["gold_lp"])
        elif it["task"] == "grounded":
            # Three scores: citation format (rules), correctness alone (judge), then
            # correctness plus support by the cited passages (judge).
            ids = {c["chunk_id"] for c in r["context"]}
            rec["cite_valid"] = citations_valid(out, ids)
            buckets["cite_valid"].append(rec["cite_valid"])
            # Every grounded item is answerable from its passages, so the abstain phrase here
            # is a false refusal. By rule; the judge's grounded_acc already scores it 0.
            rec["false_abstain"] = int(abstained(out, ABSTAIN_PHRASE))
            buckets["false_abstain"].append(rec["false_abstain"])
            if not substance(out):
                # Empty or citation-only: nothing to grade. Scored by rule, since the judge
                # passed some of these.
                rec["correct"], rec["supported"] = 0, 0
                rec["acc_reason"] = rec["judge_reason"] = "empty or citation-only answer"
                buckets["grounded_acc"].append(0)
                buckets["cite_supported"].append(0)
            elif judge:
                # Cached on the full rubric prompt: the same answer to the same item gets the
                # same verdict across runs.
                args = (r["question"], r["context"], r["gold_chunk_ids"], out)
                v = judge("grounded_acc", grounded_acc_rubric(*args))
                if v is None:
                    rec["acc_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["correct"], rec["acc_reason"] = v["score"], v["reason"]
                    buckets["grounded_acc"].append(v["score"])
                if not ids.intersection(citations(out)):
                    # Nothing cited, so nothing can be supported by a citation. Scored by rule:
                    # the judge passed uncited answers when asked to check this itself.
                    v = {"score": 0, "reason": "cites no provided passage"}
                else:
                    v = judge("grounded", grounded_rubric(*args))
                if v is None:
                    rec["judge_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["supported"], rec["judge_reason"] = v["score"], v["reason"]
                    buckets["cite_supported"].append(v["score"])
        elif it["task"] == "vocab":
            # Judge-only: definitions vary too much in wording for string matching. An output
            # with no definition line (a bare "**Term: x**" header) is 0 by rule: the judge
            # passed 193 of those, quoting definitions that weren't there.
            if not answer_line(out):
                rec["correct"], rec["judge_reason"] = 0, "no definition in the output"
                buckets["vocab_recall"].append(0)
                split("vocab", r, 0)
            elif judge:
                v = judge("vocab", vocab_rubric(r["term"], r["definition"], out))
                if v is None:
                    rec["judge_reason"], n_failed = "judge_failed", n_failed + 1
                else:
                    rec["correct"], rec["judge_reason"] = v["score"], v["reason"]
                    buckets["vocab_recall"].append(v["score"])
                    split("vocab", r, v["score"])
        elif it["task"] == "adversarial":
            # The exact abstain phrase is scored without a judge call; any other wording goes
            # to the judge, which also accepts paraphrased refusals.
            if abstained(out, ABSTAIN_PHRASE):
                rec["abstained"], rec["judge_reason"] = 1, "exact abstain phrase"
            elif not substance(out):
                # Empty or citation-only is not a refusal. By rule: the judge called 11 of 15
                # citation-only answers refusals.
                rec["abstained"], rec["judge_reason"] = 0, "empty or citation-only answer"
            elif judge:
                v = judge("adversarial", adversarial_rubric(r["question"], out))
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
    if buckets["gold_lp"]:  # a few long gold strings can dominate the mean
        metrics["gold_lp_median"] = round(statistics.median(buckets["gold_lp"]), 4)
        metrics["gold_lp_n"] = len(buckets["gold_lp"])
        ends = [it["gold_lp_end"] for it in items if it.get("gold_lp_end") is not None]
        if ends:  # the end marker's share, so the answer's own log-probability can be read apart
            metrics["gold_lp_end"] = round(statistics.fmean(ends), 4)
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
    # lm-eval rows are always chat-off (run_lm_eval.sh). A chat-template run, e.g. an old one
    # pulled back off the Modal volume, is skipped so the newest-file rule can't pick it up.
    chat_runs = [
        f for f in files if json.loads(pathlib.Path(f).read_text()).get("fewshot_as_multiturn")
    ]
    for f in chat_runs:
        print(f"skipping chat-template lm-eval results (not comparable): {f}")
    files = [f for f in files if f not in chat_runs]
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

    # lm-eval's group numbers: sample-weighted means over the subjects in each group
    metrics["mmlu"] = pick("mmlu", "acc,none")
    metrics["mmlu_stem"] = pick("mmlu_stem", "acc,none")
    metrics["mmlu_hum"] = pick("mmlu_humanities", "acc,none")
    metrics["mmlu_soc"] = pick("mmlu_social_sciences", "acc,none")
    metrics["mmlu_other"] = pick("mmlu_other", "acc,none")
    # strict-match needs the "#### <answer>" format; flexible-extract (last number) is the fallback
    metrics["gsm8k"] = pick("gsm8k", "exact_match,strict-match", "exact_match,flexible-extract")
    # acc_norm (length-normalised log-likelihood) is the standard HellaSwag number
    metrics["hellaswag"] = pick("hellaswag", "acc_norm,none", "acc,none")


def merge_ppl(metrics: dict, ppl_dir: str | None, run_name: str) -> None:
    """Pull the perplexity columns out of <ppl_dir>/<run-name>.json (eval/perplexity.py), if
    present. They live in their own file because --rescore rebuilds metrics.json from scratch."""
    if not ppl_dir:
        return
    path = pathlib.Path(ppl_dir) / f"{run_name}.json"
    if not path.exists():
        print(f"no perplexity results at {path}")
        return
    res = json.loads(path.read_text())
    for c in PPL:
        metrics[c] = res.get(c)


def item_counts(tasks_dir: pathlib.Path) -> dict[str, int]:
    """Items per task, then domain_qa items per answer kind (the qa_num / qa_ident / qa_term
    denominators), for the table's first line."""
    counts = {t: len(load_jsonl(tasks_dir / f"{t}.jsonl")) for t in TASKS}
    kinds = {"number": 0, "identifier": 0, "term": 0}
    for r in load_jsonl(tasks_dir / "domain_qa.jsonl"):
        kind = r.get("answer_kind") or answer_kind(r["answer"], r["answer_type"])
        kinds["term" if kind == "other" else kind] += 1
    return counts | {f"qa_{k}": n for k, n in kinds.items()}


def append_table(table: pathlib.Path, run_name: str, metrics: dict, items: dict[str, int]) -> None:
    """Append one row to results/table.md, writing the item-count line and header first if the
    file is new. Append-only: re-scoring a run adds a second row, so delete the stale one by hand.

    The first line records the task sizes the table was started on. A row must have scored every
    item of a task or none (--allow-partial blanks a task): a table never mixes item sets in one
    column. When a task file changes, freeze the table (results/table_vN.md) and start a new one."""
    tag = "<!-- items: " + " ".join(f"{t}={n}" for t, n in items.items()) + " -->"
    header = "| run | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 1) + "\n"
    if not table.exists():
        table.write_text(tag + "\n" + header)
    lines = table.read_text().splitlines()
    if lines[0] != tag:
        raise SystemExit(
            f"{table} was started on {lines[0]!r}, the task files now hold {tag!r}: freeze it as "
            "results/table_vN.md and start a new table"
        )
    if lines[1] != header.splitlines()[0]:
        # COLUMNS changed since the table was started; appending would misalign every cell.
        # metrics.json is already written by then, so stopping here loses nothing.
        raise SystemExit(f"{table}: header doesn't match COLUMNS; update its header row first")
    partial = {t: n for t, n in metrics["n"].items() if n not in (0, items[t])}  # tasks only
    if partial:
        raise SystemExit(f"{run_name} scored part of a task {partial} of {items}: not tabled")
    cells = [
        "" if metrics.get(c) is None else f"{metrics[c]:.2f}" if c in PPL else f"{metrics[c]:.3f}"
        for c in COLUMNS
    ]
    with table.open("a") as f:
        f.write(f"| {run_name} | " + " | ".join(cells) + " |\n")


def write_provenance(run_dir: pathlib.Path, new: dict[str, list[dict]], args) -> None:
    """generations_meta.json: per task, the run that produced its rows in generations.jsonl."""
    import vllm

    path = run_dir / "generations_meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    for t in TASKS:
        meta.setdefault(t, {"note": "generated before provenance was recorded (2026-10-04)"})
    for t, rows in new.items():
        meta[t] = {
            "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "model": args.model,
            "chat": args.chat,
            "config_format": args.config_format,
            "kv_cache_dtype": args.kv_cache_dtype,
            "vllm": vllm.__version__,
            "n": len(rows),
            "prompts_sha256": hashlib.sha256(
                "\n".join(r["prompt"] for r in rows).encode()
            ).hexdigest(),
        }
    path.write_text(json.dumps(meta, indent=2) + "\n")


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
    ap.add_argument("--ppl-dir", help="e.g. results/ppl; merges <dir>/<run-name>.json")
    ap.add_argument(
        "--seen-chunks",
        help="source chunks SFT synthesis used (default <tasks-dir>/sft_seen_chunks.txt)",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="--rescore when saved generations miss items now in a task file: that task's "
        "columns are left blank (never scored on the subset)",
    )
    ap.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="tasks to generate (comma-separated); the rest of generations.jsonl is kept",
    )
    ap.add_argument(
        "--config-format",
        default="hf",
        choices=("hf", "auto"),
        help="vLLM config format; auto only to extend Stage 0's native-path hub runs",
    )
    ap.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=("auto", "fp8"),
        help="vLLM KV cache dtype; fp8 only for Stage 6's FP8-KV variant",
    )
    ap.add_argument(
        "--gold-lp-only",
        action="store_true",
        help="recompute gold_lp for the saved domain_qa rows (no generation) and stop",
    )
    ap.add_argument(
        "--judge-cache",
        help="judge verdict cache (default <results-dir>/judge_cache.jsonl, required when "
        "--results-dir isn't the repo's results/: results/judge_cache.jsonl reuses it)",
    )
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    if unknown := set(tasks) - set(TASKS):
        ap.error(f"unknown tasks {sorted(unknown)}; choose from {TASKS}")
    if needs_chat(args.model, args.run_name) and not args.chat:
        # KPI eval always runs chat checkpoints in chat format (rule 2, notes/decisions.md).
        raise SystemExit("a chat checkpoint (Instruct, SFT/DPO/GRPO) must be evaluated with --chat")

    results = pathlib.Path(args.results_dir)
    run_dir = results / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"

    if args.gold_lp_only:
        # The gold-answer log-probabilities only: outputs, and so every accuracy column, stay as
        # saved. Used after the end-marker fix (2026-10-04).
        if not args.model:
            ap.error("--gold-lp-only needs --model")
        rows = load_jsonl(gen_path)
        qa = [it for it in build_items(pathlib.Path(args.tasks_dir), None, ["domain_qa"])]
        llm = make_llm(
            args.model,
            args.tp,
            args.max_model_len,
            args.tokenizer_mode,
            args.config_format,
            args.kv_cache_dtype,
        )
        gold_logprobs(llm, qa, args.model, args.chat)
        by_id = {it["id"]: it for it in qa}
        for r in rows:
            if r["task"] == "domain_qa" and r["id"] in by_id:
                for k in GOLD_FIELDS:
                    r.pop(k, None)
                    if k in by_id[r["id"]]:
                        r[k] = by_id[r["id"]][k]
        write_jsonl(gen_path, rows)
        import vllm

        meta_path = run_dir / "generations_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("domain_qa", {})["gold_lp"] = {
            "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "model": args.model,
            "end_marker": "EOS" if args.chat else "\\n\\n",
            "vllm": vllm.__version__,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"gold_lp rewritten for {len(qa)} domain_qa rows -> {gen_path}")
        return

    if args.rescore:
        # Re-attach saved outputs to freshly built items by (task, id). Items with no saved
        # generation are dropped, so --rescore with --limit scores that subset only.
        saved = {(r["task"], r["id"]): r for r in load_jsonl(gen_path)}
        items = build_items(pathlib.Path(args.tasks_dir), args.limit)
        missing = {}
        for it in items:
            if (it["task"], it["id"]) not in saved:
                missing[it["task"]] = missing.get(it["task"], 0) + 1
        if missing and not args.limit:
            # A task file grew after this run generated (domain_qa did, 2026-10-04). Scoring
            # the old subset would put a different item set under the same column name.
            if not args.allow_partial:
                raise SystemExit(
                    f"{gen_path} has no generation for {missing} items now in {args.tasks_dir}: "
                    "regenerate those tasks (--tasks), or --allow-partial to leave them blank"
                )
            print(f"--allow-partial: {sorted(missing)} not fully generated, columns left blank")
            items = [it for it in items if it["task"] not in missing]
        items = [it for it in items if (it["task"], it["id"]) in saved]
        for it in items:
            row = saved[(it["task"], it["id"])]
            it["output"] = row["output"]
            for k in GOLD_FIELDS:
                if k in row:
                    it[k] = row[k]
    else:
        if not args.model:
            ap.error("--model is required unless --rescore")
        items = build_items(pathlib.Path(args.tasks_dir), args.limit, tasks)
        generate(
            items,
            args.model,
            args.chat,
            args.tp,
            args.max_model_len,
            args.tokenizer_mode,
            args.config_format,
            args.kv_cache_dtype,
        )
        # Save before scoring: GPU time is the expensive part, so a judge or scoring failure
        # never costs a regeneration (fix it and --rescore). Tasks not regenerated keep their
        # saved rows, keyed by (task, id), so one file can hold generations from two dates;
        # generations_meta.json records, per task, which run produced them.
        new = {
            t: [
                {
                    "task": it["task"],
                    "id": it["id"],
                    "prompt": it["prompt_final"],
                    "output": it["output"],
                    # chat one-line tasks: the full reply before scorers.answer_line
                    **({"raw_output": it["raw_output"]} if "raw_output" in it else {}),
                    "finish_reason": it["finish_reason"],
                    **{k: it[k] for k in GOLD_FIELDS if k in it},
                }
                for it in items
                if it["task"] == t
            ]
            for t in tasks
        }
        old = [r for r in load_jsonl(gen_path) if r["task"] not in tasks]
        rows = [r for t in TASKS for r in new.get(t, old) if r["task"] == t]
        write_jsonl(gen_path, rows)
        write_provenance(run_dir, new, args)
        print(f"saved {sum(map(len, new.values()))} generations ({', '.join(tasks)}) -> {gen_path}")
        if args.generate_only:
            return

    # The judge is created only when needed, so --no-judge runs need no API key or client.
    judge = None
    if not args.no_judge:
        from judge import Judge

        repo_results = pathlib.Path(__file__).resolve().parents[1] / "results"
        if args.judge_cache:
            cache = pathlib.Path(args.judge_cache)
        elif results.resolve() == repo_results.resolve():
            cache = results / "judge_cache.jsonl"
        else:
            # A scratch --results-dir has no verdicts cached: every judged item would be paid for
            # again (50 calls, 2026-10-04). Name the cache explicitly.
            ap.error("--results-dir isn't the repo's results/: pass --judge-cache")
        judge = Judge(cache)

    seen_path = pathlib.Path(
        args.seen_chunks or pathlib.Path(args.tasks_dir) / "sft_seen_chunks.txt"
    )
    seen = set(seen_path.read_text().split()) if seen_path.exists() else None
    scored, metrics = score(items, judge, seen)
    merge_lm_eval(metrics, args.lm_eval_dir, args.run_name)
    merge_ppl(metrics, args.ppl_dir, args.run_name)
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
    if args.limit:  # a smoke subset isn't a result; the table holds full task sets only
        print(f"--limit: {results / 'table.md'} not updated")
    else:
        append_table(
            results / "table.md", args.run_name, metrics, item_counts(pathlib.Path(args.tasks_dir))
        )
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
