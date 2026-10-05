"""The SFT builder's only reader of eval/tasks/ (rule 10). Imported by the steps that select and
check (sft_pool, sft_filter, sft_assemble), never by the steps that call the LLM, so no generator
prompt can carry an eval question, answer or chunk id.

Allowed chunks (every passage of every SFT prompt, gold, neighbour or distractor):
  the seen half      eval/tasks/sft_seen_chunks.txt (forced into the pool)
  + free chunks      not in eval_chunk_ids.txt and seen by sft_split.is_seen ("seen-hash")
  - the buffer       every chunk on the same page as, or a page next to, an unseen-half source
                     chunk (domain_qa / vocab unseen, held_back, locators, few-shot): of the 215
                     unseen and held-back source chunks, 129 have a free chunk on their own page
                     and 213 on a neighbouring one, so a fact could cross a chunk boundary.
Question checks:
  rule 1   any SFT question sharing more than half its word 4-grams with one eval question
           (domain_qa, held_back, locators, few-shot, grounded, adversarial); terms of unseen vocab
           items, and VOCAB_SHOTS terms, by make_tasks.term_key
  rule 2   (ordinary chunks) an answer equal to an eval answer from the same document, scored the
           eval's way (scorers.qa_correct with the item's tolerance). The plan's "same chunk" can't
           happen for ordinary chunks, which are disjoint from eval chunks; the same document is
           where a fact is restated.
  fact_seen  which seen-half eval items an SFT answer from their own chunk matches (coverage)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# eval/ modules (prompts, scorers, ...), as eval/run_eval.py imports them
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from prompts import VOCAB_SHOTS
from scorers import normalize, qa_correct
from sft_common import ROOT, make_tasks
from sft_split import is_seen

TASKS = ROOT / "eval/tasks"
CHUNK_ID = re.compile(r"^(?P<doc>.+):p(?P<page>\d+):c(?P<n>\d+)$")


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (TASKS / f"{name}.jsonl").read_text().splitlines()]


def lines(name: str) -> set[str]:
    return set((TASKS / name).read_text().split())


def page_of(chunk_id: str) -> tuple[str, int]:
    m = CHUNK_ID.match(chunk_id)
    assert m, chunk_id
    return m["doc"], int(m["page"])


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grams4(ws: list[str]) -> set[tuple]:
    return {tuple(ws[i : i + 4]) for i in range(len(ws) - 3)}


class Guard:
    def __init__(self) -> None:
        mt = make_tasks()
        self.term_key = mt.term_key
        self.qa, self.vocab, self.fewshot = rows("domain_qa"), rows("vocab"), rows("fewshot")
        self.held = rows("held_back") + rows("locators")
        self.grounded, self.adversarial = rows("grounded"), rows("adversarial")
        self.eval_ids = lines("eval_chunk_ids.txt")
        self.seen = lines("sft_seen_chunks.txt")

        unseen = {
            i["source_chunk"] for i in self.qa + self.vocab if i["source_chunk"] not in self.seen
        }
        unseen |= {i["source_chunk"] for i in self.held + self.fewshot}
        self.unseen_sources = unseen
        self.buffer_pages = {
            (doc, page + d) for doc, page in map(page_of, unseen) for d in (-1, 0, 1)
        }

        # rule 1: every eval question, as (id, words, 4-grams)
        named = [
            (i["id"], i["question"]) for i in self.qa + self.held + self.grounded + self.adversarial
        ]
        named += [(f"fewshot-{k}", s["question"]) for k, s in enumerate(self.fewshot)]
        self.questions = [(i, words(q), grams4(words(q))) for i, q in named]
        self.blocked_terms = {
            self.term_key(i["term"]): i["id"]
            for i in self.vocab
            if i["source_chunk"] not in self.seen
        }
        self.blocked_terms.update({self.term_key(s["term"]): "vocab-shot" for s in VOCAB_SHOTS})

        # rule 2: eval answers per document; fact_seen: seen items per chunk
        self.answers_by_doc = defaultdict(list)
        for k, i in enumerate(self.qa + self.held + self.fewshot):
            self.answers_by_doc[i["source_chunk"].split(":")[0]].append(
                {**i, "id": i.get("id", f"fewshot-{k}")}
            )
        for i in self.vocab:
            self.answers_by_doc[i["source_chunk"].split(":")[0]].append(i)
        self.seen_items = defaultdict(list)
        for i in self.qa + self.vocab:
            if i["source_chunk"] in self.seen:
                self.seen_items[i["source_chunk"]].append(i)

    # ---- chunks ----
    def allowed(self, chunk_id: str) -> bool:
        if chunk_id in self.seen:
            return True
        return (
            chunk_id not in self.eval_ids
            and is_seen(chunk_id)
            and page_of(chunk_id) not in self.buffer_pages
        )

    def why_not(self, chunk_id: str) -> str:
        """The first reason a chunk is off-limits (for the pool stats)."""
        if chunk_id in self.eval_ids:
            return "eval_chunk"
        if not is_seen(chunk_id):
            return "unseen_hash"
        return "buffer"

    # ---- questions ----
    def rule1(self, question: str) -> str | None:
        """The eval question id an SFT question copies more than half of (by its word 4-grams), or
        None. Questions under 4 words are compared whole."""
        ws = words(question)
        g = grams4(ws)
        for qid, ews, eg in self.questions:
            if len(ws) < 4 or len(ews) < 4:
                if ws == ews:
                    return qid
            elif len(g & eg) > len(g) / 2:
                return qid
        return None

    def term_block(self, term: str) -> str | None:
        """An unseen vocab item's id (or "vocab-shot") if the term is one of theirs."""
        return self.blocked_terms.get(self.term_key(term))

    def rule2(self, doc: str, answer: str, term: str | None = None) -> str | None:
        """The id of an eval item from the same document whose answer this one equals."""
        for i in self.answers_by_doc.get(doc, []):
            if "term" in i:
                if term and self.term_key(term) == self.term_key(i["term"]):
                    return i["id"]
            elif qa_correct(answer, i["answer"], i["answer_type"], i.get("tolerance", 0.02)):
                return i["id"]
        return None

    def fact_seen(self, chunk_id: str, answer: str | None, term: str | None = None) -> list[str]:
        """Seen-half eval items on this chunk that the SFT answer (or term) matches."""
        out = []
        for i in self.seen_items.get(chunk_id, []):
            if "term" in i:
                if term and self.term_key(term) == self.term_key(i["term"]):
                    out.append(i["id"])
            elif answer and qa_correct(answer, i["answer"], i["answer_type"], i["tolerance"]):
                out.append(i["id"])
        return out

    def eval_questions_normalized(self) -> list[str]:
        """Every eval question, normalised (for the generator-prompt audit in the tests)."""
        return [normalize(" ".join(ws)) for _, ws, _ in self.questions]
