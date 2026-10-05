"""Shared pieces of the Stage 3 SFT data builder.

  make sft-data   # sft_pool -> sft_questions -> sft_filter -> sft_answers -> sft_judge
                  #   -> sft_replay -> sft_assemble, then eval/contamination.py --only sft

Each step reads the previous step's file in data/sft/work/ and writes its own section of
data/sft/stats.json. This module holds the paths, the Mistral client (its own disk cache, paced
under the key's 30 requests a minute), the asker personas and the paraphrased instruction wordings.

Rule 10 split: nothing here reads eval/tasks/. sft_guard.py is the only reader, and only the steps
that never call the LLM import it (sft_pool, sft_filter, sft_assemble). The steps that call it
(sft_questions, sft_answers, sft_judge) see chunk text, document titles and their own work files,
never a chunk id, a page or an eval file; tests/test_sft_data.py checks both.
"""

import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))  # prompts, scorers, judge, qa_rules, sft_split, gold_lp
from judge import JUDGE_MODEL

SFT = ROOT / "data/sft"
WORK = SFT / "work"
STATS = SFT / "stats.json"
CHUNKS = ROOT / "data/processed/chunks.jsonl"
CACHE_PATH = SFT / ".cache" / "llm_cache.jsonl"

TEACHER = JUDGE_MODEL  # Mistral Large 3 (mistral-large-2512): question writer, teacher and judge
# Mistral Medium 3.5, the dated id behind mistral-medium-latest / mistral-medium-3.5 in the API's
# model list (2026-10-04). Writes SECOND_SHARE of the completions: style diversity, and with Large
# judging every item, acceptance per teacher measures the judge's preference for its own model.
SECOND_TEACHER = "mistral-medium-2604"
SECOND_SHARE = 0.25
BASE = "mistralai/Ministral-3-8B-Base-2512"  # its Tekken tokenizer counts tokens

# The sentence GROUNDED_INSTRUCTIONS asks for; scorers.abstained matches it lowercased.
ABSTAIN_REPLY = "Not in the provided passages."

# Full-passage reads (sft_audit.py read-packets / read-merge): one verdict per record, with the
# fingerprint of what was read, so a record regenerated under the same id is read again.
READS = SFT / "read_filter.jsonl"
# Documents whose records were regenerated after a metadata fix: none of their records is built
# from a read made before it (fema-p-2355 is FEMA P-2335 on its own cover; sources.csv said P-2355
# until 2026-10-05).
REGENERATED_DOCS = {"fema-p-2355"}


def fingerprint(e: dict) -> str:
    """What a read verdict vouches for: the question (or term), the answer and the passages shown."""
    passages = [p["chunk_id"] for p in e.get("passages") or []]
    payload = json.dumps([e.get("question"), e.get("term"), e.get("answer"), passages])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ files ---
def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} -> {path.relative_to(ROOT)}")
    return len(rows)


def update_stats(section: str, data: dict) -> None:
    """Each step owns one top-level key of data/sft/stats.json; re-running it rewrites only that."""
    stats = json.loads(STATS.read_text()) if STATS.exists() else {}
    stats[section] = data
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")


def load_chunks() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """data/processed/chunks.jsonl by id, and per document in reading order (the file's order), each
    chunk carrying its position `pos` in that order."""
    by_id, by_doc = {}, defaultdict(list)
    for c in read_jsonl(CHUNKS):
        c["pos"] = len(by_doc[c["doc"]])
        by_doc[c["doc"]].append(c)
        by_id[c["chunk_id"]] = c
    return by_id, dict(by_doc)


def doc_titles() -> dict[str, str]:
    """Document title as the generator sees it, from data/sources.csv. USACE rows are
    "EM 1110-2-2906 | CECW-ED | Design of Pile Foundations | 1/15/1991"; they become
    "EM 1110-2-2906: Design of Pile Foundations (1991)", so questions can name the manual."""
    out = {}
    with open(ROOT / "data/sources.csv") as f:
        for r in csv.DictReader(f):
            parts = [p.strip() for p in r["title"].split("|")]
            if len(parts) == 4:
                out[r["slug"]] = f"{parts[0]}: {parts[2]} ({parts[3][-4:]})"
            else:
                out[r["slug"]] = r["title"].strip()
    return out


def h01(key: str) -> float:
    """A stable pseudo-random number in [0, 1) per key: assignments don't move when the pool grows
    (the pilot's chunks keep their wording, persona and teacher in the full run)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def pick(key: str, options: list):
    return options[int(h01(key) * len(options))]


def quotas(weights: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder apportionment of n over the keys, proportional to the weights."""
    total = sum(weights.values())
    exact = {k: n * w / total for k, w in weights.items()}
    q = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - q[k], reverse=True)[: n - sum(q.values())]:
        q[k] += 1
    return q


STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by", "as", "at",
    "from", "is", "are", "be", "was", "were", "that", "this", "which", "what", "when", "how",
    "why", "where", "who", "does", "do", "it", "its", "into", "than", "then", "there", "their",
    "these", "those", "per", "under", "over", "not", "no", "can", "may", "shall", "should",
    "must",
}  # fmt: skip


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in STOPWORDS]


def gold_present(gold: str, text: str) -> bool:
    """Whether a passage states the gold answer, even partly: the hard rule for abstain items.
    A short gold (6 words or fewer) counts as present when its words appear together in the passage,
    or any of its numbers with two or more significant characters ("50", "0.75") does, by value; a
    longer gold (a procedure's gist) when 60% of its content words do."""
    norm = lambda t: " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower().replace(",", "")))
    g, t = norm(gold), norm(text)
    if not g:
        return False
    if len(g.split()) <= 6:
        if f" {g} " in f" {t} ":
            return True
        values = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", t)}
        nums = [n for n in re.findall(r"\d+(?:\.\d+)?", g) if len(n.replace(".", "")) >= 2]
        return any(float(n) in values for n in nums)
    words = set(content_words(gold))
    return bool(words) and len(words & set(content_words(text))) / len(words) >= 0.6


def teacher_for(key: str) -> str:
    return SECOND_TEACHER if h01(f"sft-teacher:{key}") < SECOND_SHARE else TEACHER


# ------------------------------------------------------------- the client ---
class Pacer:
    """Client-side pacing shared by every worker: calls start at least `interval` seconds apart.
    A copy of make_tasks.Pacer (importing make_tasks builds its own client and loads the eval's
    call cache): pinned just under the key's limit, a 429 stretches it x1.25 at most once per 10 s,
    10 successes in a row shrink it back toward the floor."""

    def __init__(self, interval: float, floor: float, ceiling: float = 4.0):
        self.interval, self.floor, self.ceiling = interval, floor, ceiling
        self.next_start, self.ok, self.last_slow, self.calls = 0.0, 0, 0.0, 0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        time.sleep(start - now)

    def result(self, throttled: bool) -> None:
        with self.lock:
            now = time.monotonic()
            if throttled:
                self.ok = 0
                if now - self.last_slow > 10:
                    self.interval, self.last_slow = min(self.ceiling, self.interval * 1.25), now
            else:
                self.ok, self.calls = self.ok + 1, self.calls + 1
                if self.calls % 50 == 0:
                    print(
                        f"  pacer: {self.calls} calls, interval {self.interval:.2f} s", flush=True
                    )
                if self.ok >= 10:
                    self.interval, self.ok = max(self.floor, self.interval * 0.9), 0


RPM = 30  # the key's limit, shared by every model (checked 2026-10-04); --rpm overrides
PACER = Pacer(interval=60 / RPM * 1.03, floor=60 / RPM * 1.03)
FAILED_CALLS = 0
_client = None
_cache: dict[str, dict] | None = None
_cache_lock = threading.Lock()


def set_rpm(rpm: float) -> None:
    PACER.interval = PACER.floor = 60 / rpm * 1.03


def _load_cache() -> dict[str, dict]:
    """The cache, loaded once under the lock: pmap's workers make their first call together, and an
    unlocked load gave each its own dict, so they re-asked prompts another had just answered (33
    keys were written twice, with different outputs, by 2026-10-05). The first answer for a key
    wins, so a cached prompt always returns the same output."""
    global _cache
    with _cache_lock:
        if _cache is None:
            cache: dict[str, dict] = {}
            if CACHE_PATH.exists():
                for line in CACHE_PATH.open():
                    rec = json.loads(line)
                    cache.setdefault(rec["key"], rec["out"])
            _cache = cache
    return _cache


def llm_json(
    prompt: str, step: str, model: str = TEACHER, temperature: float = 0.2, retries: int = 6
) -> dict:
    """One call in JSON mode through the disk cache (data/sft/.cache/llm_cache.jsonl, gitignored).
    Keyed on (model, temperature, prompt); each record also keeps the step and the prompt, so the
    prompts the generator saw can be audited (tests/test_sft_data.py). Returns {} when every retry
    failed; failures are never cached, so a rerun retries them."""
    global _client, FAILED_CALLS
    cache = _load_cache()
    key = hashlib.sha256(json.dumps([model, temperature, prompt]).encode()).hexdigest()
    if key in cache:
        return cache[key]
    if _client is None:
        from mistralai.client import Mistral

        _client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=120_000)
    for attempt in range(retries):
        PACER.wait()
        try:
            r = _client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            message = r.choices[0].message
            content = message.content if message else None
            if not isinstance(content, str):
                raise TypeError(f"model returned no text content: {content!r}")
            out = json.loads(content)
            if not isinstance(out, dict) or not out:
                raise TypeError(f"expected a JSON object, got {content[:100]!r}")
            PACER.result(throttled=False)
            break
        except Exception as e:  # noqa: BLE001  any failure (rate limit, network, bad JSON) is retried
            print(f"  retry {attempt + 1} ({step}): {str(e)[:160]}", flush=True)
            if "429" in str(e):
                PACER.result(throttled=True)
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(2**attempt)
    else:
        FAILED_CALLS += 1
        return {}
    rec = {"key": key, "step": step, "model": model, "temperature": temperature}
    with _cache_lock:
        if key in cache:  # another worker answered the same prompt meanwhile: keep the first
            return cache[key]
        cache[key] = out
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a") as f:
            f.write(json.dumps({**rec, "prompt": prompt, "out": out}, ensure_ascii=False) + "\n")
    return out


def report_failures() -> None:
    if FAILED_CALLS:
        print(f"WARNING: {FAILED_CALLS} calls failed after every retry; rerun to retry them")


WORKERS = 8


def pmap(fn, items, workers: int = WORKERS) -> list:
    """Parallel map over threads (network-bound); results in input order."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def make_tasks():
    """eval/make_tasks.py, for its pure filters (CONTEXT_BOUND, is_trivia, term_key, ...). Its
    import builds an API client from MISTRAL_API_KEY and reads the eval's call cache; neither is
    used here, so a placeholder key is fine when none is loaded (tests)."""
    os.environ.setdefault("MISTRAL_API_KEY", "unset")
    import make_tasks

    return make_tasks


# --------------------------------------------------------------- wording ---
# Who asks, for the half of each format that is paraphrased (A2's {asker}); the exact half uses
# NEUTRAL_ASKER with the eval's instruction text word for word.
NEUTRAL_ASKER = "a practicing structural or civil engineer"
PERSONAS = [
    "a municipal plan reviewer",
    "a state DOT bridge designer",
    "a bridge inspector",
    "a USACE dam safety engineer",
    "a FEMA mitigation planner",
    "a graduate student in structural engineering",
    "an engineer-in-training preparing for the PE exam",
    "a consulting structural engineer",
    "a building official",
    "a contractor's quality-control manager",
    "a forensic engineer investigating a failure",
    "a NASA structures analyst",
]

# Paraphrased instructions (wording "paraphrased"). Each keeps what the eval scores: a bare answer
# for closed-book, one sentence for definitions, [id] citations and the exact abstain sentence for
# the passage tasks. None repeats the eval's own sentences.
QA_TEMPLATES = [
    "{q} Give the answer only.",
    "{q}\nReply with just the value or name, no explanation.",
    "Short answer only, please: {q}",
    (
        "You answer questions about US federal structural engineering documents (USACE, FHWA, "
        "NIST, FEMA, NASA) with the bare answer.\n\nQuestion: {q}"
    ),
]
DEF_TEMPLATES = [
    'In structural engineering, what does "{t}" mean? Answer in one sentence.',
    'Define "{t}" in one sentence.',
    "What is meant by {t}? One precise sentence, please.",
    'Give a one-sentence technical definition of "{t}".',
]
# (instruction around {passages} and {q}, one passage's layout)
RAG_TEMPLATES = [
    (
        (
            "Answer using only the passages below. Cite the passage id in brackets after each claim. "
            "If the passages do not contain the answer, reply exactly: Not in the provided passages."
            "\n\n{passages}\n\nQuestion: {q}"
        ),
        "[{id}] {text}",
    ),
    (
        (
            "Use only the following passages to answer. Put the id of the supporting passage in square "
            "brackets after every claim. If the answer is not in them, reply exactly: Not in the "
            "provided passages.\n\nPassages:\n\n{passages}\n\n{q}"
        ),
        "[{id}]\n{text}",
    ),
    (
        (
            "{q}\n\nBase your answer only on these passages and cite each claim with its passage id in "
            "square brackets. If they don't contain the answer, reply exactly: Not in the provided "
            "passages.\n\n{passages}"
        ),
        "[{id}]\n{text}",
    ),
    (
        (
            "Here are passages from engineering documents.\n\n{passages}\n\nUsing only these passages, "
            "answer: {q}\nCite the passage id in square brackets after each claim. If the answer isn't "
            "there, reply exactly: Not in the provided passages."
        ),
        "[{id}] {text}",
    ),
]
MULTI_STEP_SUFFIX = (
    '\nShow the calculation, then give the result on a last line starting with "Answer:".'
)
