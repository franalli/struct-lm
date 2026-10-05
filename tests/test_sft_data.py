"""The Stage 3 SFT set (data/sft/) against rule 10 and the trainer's format, checked on the frozen
files, plus the frozen v2 eval it was built against. The SFT checks are skipped until
data/sft/train.jsonl exists; the generator-prompt audit also needs the builder's call cache
(data/sft/.cache, gitignored)."""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data/scripts"))
SFT = REPO / "data/sft"
TASKS = REPO / "eval/tasks"
CACHE = SFT / ".cache/llm_cache.jsonl"
CHUNK_ID = re.compile(r"[\w.-]+:p\d+:c\d+\b")
GENERATORS = ("sft_common.py", "sft_questions.py", "sft_answers.py", "sft_judge.py")
FORMATS = {"closed_book", "definition", "grounded", "abstain", "replay"}
KEYS = {
    "prompt", "completion", "format", "kind", "source_chunks", "distractors", "split", "persona",
    "wording", "paraphrase_of", "teacher", "rubric", "origin", "question", "fact_id", "fact_seen",
    "final_answer", "replay_source", "eid", "n_tokens",
}  # fmt: skip

HAVE = pytest.mark.skipif(
    not (SFT / "train.jsonl").exists(), reason="data/sft/train.jsonl not built yet"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sums(path: Path) -> dict[str, str]:
    return {name: h for h, name in (line.split() for line in path.read_text().splitlines())}


def test_eval_v2_frozen():
    """The eval the SFT set was decontaminated against (rule 9): v2, 2026-10-04."""
    for name, h in sums(TASKS / "SHA256SUMS").items():
        assert sha(TASKS / name) == h, name


@pytest.fixture(scope="module")
def records() -> list[dict]:
    out = []
    for name, split in (("train.jsonl", "train"), ("sft_val.jsonl", "val")):
        out += [{**json.loads(line), "_file": split} for line in (SFT / name).open()]
    return out


@pytest.fixture(scope="module")
def guard():
    from sft_guard import Guard

    return Guard()


@HAVE
def test_sft_frozen():
    for name, h in sums(SFT / "SHA256SUMS").items():
        assert sha(SFT / name) == h, name


@HAVE
def test_schema(records):
    for r in records:
        assert set(r) - {"_file"} == KEYS, r["eid"]
        assert [m["role"] for m in r["prompt"]] == ["user"], r["eid"]
        assert [m["role"] for m in r["completion"]] == ["assistant"], r["eid"]
        assert r["prompt"][0]["content"].strip() and r["completion"][0]["content"].strip()
        assert r["format"] in FORMATS
        assert (r["split"] is None) == (r["format"] == "replay")


@HAVE
def test_chunks_allowed(records, guard):
    """Rule 10: every passage and source chunk is the seen half or a free seen-hash chunk outside
    the unseen buffer; none is any other chunk in eval_chunk_ids.txt."""
    forbidden = guard.eval_ids - guard.seen
    for r in records:
        ids = set(r["source_chunks"]) | set(r["distractors"])
        if r["fact_id"]:
            ids.add(r["fact_id"].rsplit(":", 1)[0])
        assert not ids & forbidden, r["eid"]
        assert all(guard.allowed(i) for i in ids), r["eid"]


@HAVE
def test_eval_seen_only_in_train(records):
    assert not [r["eid"] for r in records if r["origin"] == "eval_seen" and r["_file"] == "val"]


@HAVE
def test_no_eval_question_reused(records, guard):
    """Rule 1 holds on the final questions (A4's paraphrases and problems included)."""
    for r in records:
        if r["format"] == "definition":
            assert guard.term_block(r["question"]) is None, r["eid"]
        elif r["question"]:
            assert guard.rule1(r["question"]) is None, r["eid"]


@HAVE
def test_exact_prompts(records, guard):
    """wording "exact" is the eval's own prompt text, as run_eval sends it."""
    from prompts import GROUNDED_FORMAT, GROUNDED_INSTRUCTIONS, qa_prompt, vocab_prompt

    for r in (r for r in records if r["wording"] == "exact"):
        p, q = r["prompt"][0]["content"], r["question"]
        if r["format"] == "closed_book":
            assert p == qa_prompt(q, guard.fewshot), r["eid"]
        elif r["format"] == "definition":
            assert p == vocab_prompt(q), r["eid"]
        else:
            assert p.startswith(f"{GROUNDED_INSTRUCTIONS}\n{GROUNDED_FORMAT}\n\n"), r["eid"]
            assert p.endswith(f"\n\nQuestion: {q}\nAnswer:"), r["eid"]
            for i in r["source_chunks"] + r["distractors"]:
                assert f"[{i}]\n" in p, r["eid"]


@HAVE
def test_completions(records):
    from scorers import citations, citations_valid
    from sft_common import ABSTAIN_REPLY

    for r in records:
        c = r["completion"][0]["content"]
        if r["format"] == "abstain":
            assert c == ABSTAIN_REPLY and r["source_chunks"] == [], r["eid"]
            assert len(r["distractors"]) == 4, r["eid"]
        elif r["format"] == "grounded":
            ids = set(r["source_chunks"]) | set(r["distractors"])
            assert len(ids) == 4 and citations_valid(c, ids), r["eid"]
            assert set(citations(c)) == set(r["source_chunks"]), r["eid"]
        elif r["kind"] == "multi_step":
            assert re.search(r"^\s*answer\s*:", c, re.IGNORECASE | re.MULTILINE), r["eid"]


@HAVE
def test_tokens(records):
    """Every record fits the 4,096-token window in the eval's chat rendering (mistral-common,
    no system prompt), and its prompt's tokens are a prefix of the whole."""
    from gold_lp import token_ids
    from sft_common import BASE

    for r in records:
        enc = token_ids(BASE, True, r["prompt"][0]["content"], r["completion"][0]["content"])
        assert enc is not None, r["eid"]
        assert len(enc[0]) == r["n_tokens"] <= 4096, r["eid"]


@HAVE
def test_closed_book_filtered(records):
    """Every closed-book record was read against its passage (sft_audit.py filter-merge) and none
    the reader called a defect is in the set."""
    verdicts = {}
    for line in (SFT / "closed_book_filter.jsonl").open():
        v = json.loads(line)
        verdicts[v["eid"]] = v["verdict"]
    for r in records:
        if r["format"] == "closed_book":
            assert verdicts.get(r["eid"]) in ("ok", "minor"), r["eid"]
    assert not [r["eid"] for r in records if r["kind"] == "multi_step"]


def test_generators_never_read_eval():
    """The steps that call the LLM can't reach eval/tasks/: they never import sft_guard."""
    for name in GENERATORS:
        src = (REPO / "data/scripts" / name).read_text()
        assert not re.search(r"^\s*(from|import) sft_guard\b", src, re.MULTILINE), name
        assert not re.search(r"[\"']eval/tasks|/ \"tasks\"", src), name


@pytest.mark.skipif(not CACHE.exists(), reason="needs the builder's call cache (data/sft/.cache)")
def test_generator_prompts_clean(guard):
    """No prompt the builder sent (questions, answers, paraphrases, problems, judge, revise)
    carries a chunk id or an eval question."""
    evalq = [" ".join(ws) for _, ws, _ in guard.questions if len(ws) >= 6]
    for line in CACHE.open():
        rec = json.loads(line)
        p = rec["prompt"]
        assert not CHUNK_ID.search(p), (rec["step"], CHUNK_ID.search(p))
        flat = " ".join(re.findall(r"[a-z0-9]+", p.lower()))
        hits = [q for q in evalq if q in flat]
        assert not hits, (rec["step"], hits[:1])
