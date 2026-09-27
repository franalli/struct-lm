"""AI-as-judge: Mistral Large 3 (JUDGE_MODEL), temperature 0, fixed rubrics, disk cache.

Every verdict is cached on a hash of (judge model, rubric name, full rubric prompt), so
re-scoring a run is free and deterministic, two runs that produce the same answer get the
same verdict, and editing a rubric re-judges everything it grades. Failed calls are never
cached. The judge never sees which model produced the answer.

Used by run_eval.score() for the three metrics that string matching can't measure:
  grounded     cite_supported   correct, and every claim backed by a cited passage
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

JUDGE_MODEL = "mistral-large-2512"  # Mistral Large 3, pinned; must match make_tasks.GEN_MODEL


class Judge:
    """Callable judge with an append-only JSONL cache (results/judge_cache.jsonl).

    The cache is committed with the results, so anyone can `run_eval.py --rescore` a
    committed run and get identical numbers without an API key: every lookup hits.
    `calls` counts successful cache misses (real API calls); `failures` counts answers the judge
    couldn't grade after all retries, with the most recent error in `last_error`."""

    def __init__(self, cache_path: str | os.PathLike):
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

    def __call__(self, rubric: str, prompt: str) -> dict | None:
        """Return {"score": 0|1, "reason": str} for one answer, or None if the judge failed.

        rubric  name of the rubric ("grounded", "vocab", "adversarial"), part of the cache key
        prompt  the full rubric text sent to the judge (one of the *_rubric functions below).
                It already contains everything being graded (question, passages, reference,
                answer), so keying on it means a verdict is reused only for the identical
                prompt: rewording a rubric invalidates its old verdicts automatically.

        Retries up to 5 times with exponential backoff (1, 2, 4, 8, 16 s). If every attempt
        fails, returns None and caches nothing, so the next rescore retries that item. The
        caller leaves failed items out of the metric (see run_eval.score) rather than
        counting them as 0 against the model."""
        key = hashlib.sha256(json.dumps([JUDGE_MODEL, rubric, prompt]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for attempt in range(5):
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
    gold so it checks correctness against the right source rather than a distractor. Citation
    ids are not re-checked here; that is scorers.citations_valid (cite_valid)."""
    return f"""You are grading an answer produced from a set of passages. Be strict.

Passages:
{_passages(context)}

Gold passage id(s) that contain the answer: {", ".join(gold_ids)}

Question: {question}

Answer to grade:
{answer}

Score 1 only if ALL of the following hold:
1. The answer is correct according to the gold passage(s).
2. Every factual claim in the answer is supported by a passage the answer cites in square brackets.
3. The answer does not add claims that are not in the passages.
Otherwise score 0.

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


def adversarial_rubric(question: str, context: list[dict], answer: str) -> str:
    """halluc_rate: 1 means the model declined (any wording). The judge is told up front that the
    passages don't contain the answer, so it grades refusal rather than re-deciding whether
    the question is answerable (make_tasks.is_answerable already checked that)."""
    return f"""The passages below do NOT contain the answer to the question. You are checking whether
the answer correctly declines.

Passages:
{_passages(context)}

Question: {question}

Answer to grade:
{answer}

Score 1 if the answer states that the information is not available in the passages (any wording),
without asserting a specific value, clause, procedure or fact as the answer.
Score 0 if the answer asserts an answer, guesses, or presents general knowledge as if it came from
the passages.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""
