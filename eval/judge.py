"""AI-as-judge: Mistral Large 3 (JUDGE_MODEL), temperature 0, fixed rubrics, disk cache.

Every verdict is cached on a hash of (judge model, rubric name, full rubric prompt), so
re-scoring a run is free and deterministic, two runs that produce the same answer get the
same verdict, and editing a rubric re-judges everything it grades. Failed calls are never
cached. The judge never sees which model produced the answer.

Used by run_eval.score() for the four metrics that string matching can't measure:
  grounded_acc grounded_acc     correct according to the gold passage, citations ignored
  grounded     cite_supported   correct, and every claim backed by a cited passage (answers
                                citing no provided passage score 0 without a judge call)
  vocab        vocab_recall     definition consistent with the reference
  adversarial  halluc_rate      the answer declines instead of asserting (only when the exact
                                abstain phrase is missing)

Every rubric returns a binary score (0/1) and a one-sentence reason. Binary rather than a
1-5 scale because binary verdicts are more consistent between judge calls and average
directly into a rate.
"""

import hashlib
import json
import os
import pathlib
import time

from scorers import citations

JUDGE_MODEL = "mistral-large-2512"  # Mistral Large 3, pinned; must match make_tasks.GEN_MODEL


class Judge:
    """Callable judge with an append-only JSONL cache (results/judge_cache.jsonl).

    The cache is committed with the results, so anyone can `run_eval.py --rescore` a
    committed run and get identical numbers without an API key: every lookup hits.
    `calls` counts successful cache misses (real API calls); `failures` counts answers the judge
    couldn't grade after all retries, with the most recent error in `last_error`.

    Paced: API calls start at least 60 / rpm * 1.03 s apart, just under the key's limit (30 a
    minute, shared with make_tasks.py and the SFT builder), as api_eval.py paces. Without it a
    rescore bursts into the limit and spends its retries on 429s (8 verdicts failed at Stage 0)."""

    def __init__(self, cache_path: str | os.PathLike, rpm: int = 30):
        from mistralai.client import Mistral  # lazy, so --no-judge runs need no API client

        self.client = Mistral(
            api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=120_000
        )  # no default timeout: one dead socket hung a run for hours
        self.cache_path = pathlib.Path(cache_path)
        # key -> verdict. Later lines win if a key repeats (they don't in normal use).
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["verdict"]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.failures = 0
        self.last_error = ""
        self.interval, self.next_start = 60 / rpm * 1.03, 0.0

    def __call__(self, rubric: str, prompt: str) -> dict | None:
        """Return {"score": 0|1, "reason": str} for one answer, or None if the judge failed.

        rubric  name of the rubric ("grounded", "vocab", "adversarial"), part of the cache key
        prompt  the full rubric text sent to the judge (one of the *_rubric functions below).
                It already contains everything being graded (question, passages, reference,
                answer), so keying on it means a verdict is reused only for the identical
                prompt: rewording a rubric invalidates its old verdicts automatically.

        Up to 7 attempts with exponential backoff between them (1, 2, ... 64 s, ~2 min in all).
        If every attempt fails, returns None and caches nothing, so the next rescore retries that item. The
        caller leaves failed items out of the metric (see run_eval.score) rather than
        counting them as 0 against the model."""
        key = hashlib.sha256(json.dumps([JUDGE_MODEL, rubric, prompt]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for attempt in range(7):  # backoff totals ~127 s, past a per-minute rate-limit window
            time.sleep(max(0.0, self.next_start - time.monotonic()))
            self.next_start = time.monotonic() + self.interval
            try:
                r = self.client.chat.complete(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},  # guarantees parseable JSON output
                )
                message = r.choices[0].message
                content = message.content if message else None
                if not isinstance(content, str):  # empty or non-text reply: retry like any error
                    raise TypeError(f"judge returned no text content: {content!r}")
                out = json.loads(content)
                # Coerce defensively: a missing score counts as 0 (fail), the reason is capped
                # so one verbose verdict can't bloat scored.jsonl.
                verdict = {
                    "score": int(out.get("score", 0)),
                    "reason": str(out.get("reason", ""))[:300],
                }
                break
            except Exception as e:  # noqa: BLE001  any failure (rate limit, network, bad JSON) is retried
                self.last_error = str(e)[:300]
                time.sleep(2**attempt)
        else:  # all attempts failed: nothing cached, so the item is retried on the next rescore
            self.failures += 1
            return None
        self.calls += 1
        self.cache[key] = verdict
        # Appended immediately, so a crash mid-run keeps every verdict paid for so far.
        with self.cache_path.open("a") as f:
            f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
        return verdict


# ------------------------------------------------------------------ rubrics ---
def _passages(context: list[dict]) -> str:
    """Passages in the same "[chunk_id]\\ntext" layout the model saw (prompts.grounded_prompt),
    so the judge can resolve the model's citations."""
    return "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in context)


def grounded_rubric(question: str, context: list[dict], gold_ids: list[str], answer: str) -> str:
    """cite_supported: all three conditions must hold for a 1. The judge is told which passage is
    gold so it checks correctness against the right source rather than a distractor.

    The cited ids are parsed here and listed for the judge: left to find them itself, it
    passed uncited answers and quoted citations they didn't contain. run_eval scores answers
    citing no provided passage as 0 before calling the judge. Whether every bracket is a
    provided id is scorers.citations_valid (cite_valid), not re-checked here."""
    ids = {c["chunk_id"] for c in context}
    cited = [c for c in dict.fromkeys(citations(answer)) if c in ids]
    return f"""You are grading an answer produced from a set of passages. Be strict.

Passages:
{_passages(context)}

Gold passage id(s) that contain the answer: {", ".join(gold_ids)}

Question: {question}

Answer to grade:
{answer}

Passage ids the answer cites (parsed from its square brackets): {", ".join(cited) or "none"}

Score 1 only if ALL of the following hold:
1. The answer is correct according to the gold passage(s).
2. Every factual claim in the answer is supported by one of the cited passages listed above.
3. The answer does not add claims that are not in the passages.
Otherwise score 0.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""


def grounded_acc_rubric(
    question: str, context: list[dict], gold_ids: list[str], answer: str
) -> str:
    """grounded_acc: grounded_rubric with the citation condition removed, so the gap between the
    two metrics is citation behaviour alone. Base models often answer correctly without the
    [id] format; this is what separates "doesn't know" from "doesn't cite". Every grounded item
    is answerable, so declining scores 0."""
    return f"""You are grading an answer produced from a set of passages. Be strict about content, and
ignore citations entirely.

Passages:
{_passages(context)}

Gold passage id(s) that contain the answer: {", ".join(gold_ids)}

Question: {question}

Answer to grade:
{answer}

Score 1 only if ALL of the following hold:
1. The answer is correct according to the gold passage(s) and gives what the question asks for.
2. The answer does not add claims that are not in the passages or that contradict them.
Citations do not matter: an answer with no citations, or with wrong citation ids, can still score 1.
Score 0 if the answer is wrong, incomplete, declines to answer, or adds unsupported claims.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""


def vocab_rubric(term: str, reference: str, answer: str) -> str:
    """vocab_recall: the definition must describe the same concept as the reference (extracted
    from the corpus passage by make_tasks.py) without errors. Paraphrase and brevity are fine;
    vagueness is not, so a definition that would fit many terms scores 0."""
    return f"""You are grading a one-sentence definition of a structural engineering term.

Term: {term}
Reference definition: {reference}
Definition to grade: {answer}

Score 1 if the definition to grade is technically consistent with the reference (same concept, no
contradicting or fabricated detail), even if worded differently or shorter. Score 0 if it defines a
different concept, is vague enough to fit many terms, or contains a technical error.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""


def adversarial_rubric(question: str, answer: str) -> str:
    """halluc_rate: 1 means the model declined (any wording). The passages don't contain the
    answer (make_tasks.is_answerable checked that), so any answer that isn't a refusal is a
    hallucination, and only the answer's wording needs grading.

    The passages are deliberately left out. Shown them and told they lack the answer, the
    judge graded the passages instead of the answer: Base's "the maximum height difference is
    10 feet" scored as a refusal, reason "the passages do not provide any information"."""
    return f"""You are classifying whether an answer declines to answer a question. Judge only the
answer's wording; do not try to answer the question yourself or decide whether it is answerable.

Question: {question}

Answer to classify:
{answer}

Score 1 if the answer says the information is not available, not provided or not specified (any
wording). It may also mention related information, as long as it gives no answer to the question.
Score 0 if the answer gives an answer to the question (a value, a section or clause, a procedure, a
comparison or an explanation), even if hedged or next to a statement that the information is not
provided. Also score 0 if the answer is empty or consists only of citations.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""


def pairwise_rubric(prompt: str, reference: str, answer_1: str, answer_2: str) -> str:
    """Stage 4's win rate (eval/winrate.py): which of two anonymous answers to the same prompt is
    better. Kept on the binary contract: score 1 means answer 1. Asked in both orders by the
    caller, so position bias shows as a split (scored a tie). Reported only: this judge missed
    72% of known grounded defects in the Stage 4 benchmark (notes/decisions.md, 2026-10-08)."""
    return f"""You are comparing two answers to the same request. Be strict.

Request (exactly as the model saw it):
{prompt}

Reference (what a correct answer must agree with): {reference}

Answer 1:
{answer_1}

Answer 2:
{answer_2}

Decide which answer is better, in this order of importance:
1. Correct and faithful to the reference and to any passages in the request; adds nothing they don't say.
2. Does what the request asks: its format, its citations, or declining when the request says to.
3. Concise.

Return JSON: {{"score": 1 if Answer 1 is better, 0 if Answer 2 is better, "reason": "one sentence"}}"""
