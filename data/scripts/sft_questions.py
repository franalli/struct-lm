"""A2: questions from every pool chunk (Self-Instruct step) -> data/sft/work/questions.jsonl

  set -a; . ./.env; set +a
  .venv/bin/python data/scripts/sft_questions.py

One TEACHER call per chunk at temperature 0.7 (variety; the cache fixes each prompt's output), with
the document title and the passage text only: no chunk id, no page, nothing from eval/tasks/.
  ordinary   5 questions {asker} would ask that the passage answers: 2 values, 1 which-document-or-
             article, 1 definition, 1 procedure. Each comes with its gold fact (the answer A4's
             teacher answer is verified against) and becomes, after A3: closed-book (value,
             document), definition, grounded (procedure) or abstain (a value or procedure question
             paired with passages that lack the answer).
  eval_seen  exhaustive extraction: every value, identifier and term the passage states, one
             question per fact, at most 6, with a neutral asker. Nothing tells the generator which
             facts the eval asks about; sft_assemble measures how many it found (fact_seen).
"""

import argparse
from collections import Counter

from sft_common import (
    TEACHER,
    WORK,
    llm_json,
    pmap,
    read_jsonl,
    report_failures,
    set_rpm,
    update_stats,
    write_jsonl,
)

CLOSED_BOOK_RULES = """Rules for "value" and "document" questions, which are asked WITHOUT the passage:
- Name the document, structure type, load condition, material or quantity, so the question pins
  down exactly one fact on its own.
- Never refer to what the reader can't see: no "the figure/table/equation/example/passage/this
  document", no values computed or assumed in one worked example, no relative time ("currently").
- Never ask for a page, figure, table or equation number.
- It must not be answerable by general engineering common sense; it must need this document.
- The answer is at most 6 words, as stated in the passage; numbers include their unit."""


def ordinary_prompt(c: dict) -> str:
    return f"""You are writing practice questions about a US federal structural engineering document.

Document: {c["title"]}
Passage:
{c["text"]}

Write 5 questions {c["asker"]} would ask that this passage answers:
- 2 "value" questions: the answer is one number with its unit, or a short quantity (ratio,
  percentage, range), stated in the passage.
- 1 "document" question: which document, standard, article, section or clause governs or covers
  something; the answer is that designation as the passage writes it (e.g. "EM 1110-2-2104",
  "AASHTO LRFD Article 6.10.8.2"). Skip it if the passage cites none.
- 1 "definition": a technical term the passage defines or uses in a specific technical sense. Give
  the term itself as "question" and a one-sentence definition faithful to the passage as "answer".
- 1 "procedure" question: how something is done, checked, limited or justified; the answer takes
  one to three sentences from the passage. Phrase it as a standalone question (never "according to
  the passage").
Vary the verbs and the sentence structure across the questions.

{CLOSED_BOOK_RULES}

Return fewer items, or none, rather than inventing: skip any kind the passage can't support
(worked calculations, tables of contents, boilerplate and reference lists support none).

Return JSON: {{"items": [{{"kind": "value" | "document" | "definition" | "procedure", "question": "...", "answer": "..."}}]}}"""


def forced_prompt(c: dict) -> str:
    return f"""You are extracting what a US federal structural engineering document states, to write study questions.

Document: {c["title"]}
Passage:
{c["text"]}

List, as two separate lists:
1. "facts": every value and identifier this passage states, at most 6:
   - "value": a number with its unit, or a short quantity (ratio, percentage, range);
   - "identifier": a document, standard, article, section or clause designation the passage gives
     for something.
2. "terms": every domain-specific technical term the passage defines or uses in a specific
   technical sense, at most 6: plain engineering terms the passage explains or relies on as well as
   long compound names. Exclude generic words, units, organisation names and document numbers.
   Give each a one-sentence definition faithful to how the passage uses it.
For each fact and term write one question {c["asker"]} would ask whose answer is exactly that fact
or term (for a term: a question that describes it and asks for the term).

{CLOSED_BOOK_RULES}

Skip headings, page furniture, boilerplate, and values that only mean something inside one worked
example.

Return JSON: {{"facts": [{{"kind": "value" | "identifier", "answer": "...", "what": "what it is, in a few words", "question": "..."}}],
 "terms": [{{"term": "...", "definition": "...", "question": "..."}}]}}"""


KIND = {
    "value": "value",
    "document": "identifier",
    "identifier": "identifier",
    "definition": "term",
    "term": "term",
    "procedure": "procedure",
}


def questions(c: dict) -> list[dict]:
    """The chunk's items, normalised to one schema: {qid, kind, question, answer, term, definition,
    what}. Ordinary "definition" items carry only term + definition (no closed-book question);
    forced term facts carry both a question asking for the term and its definition."""
    forced = c["origin"] == "eval_seen"
    out = llm_json(
        forced_prompt(c) if forced else ordinary_prompt(c), step="questions", temperature=0.7
    )
    if forced:  # facts f0-f5, terms f6-f11 (fact ids stay "<chunk>:f<n>")
        facts = [i for i in out.get("facts", []) if isinstance(i, dict)][:6]
        terms = [i for i in out.get("terms", []) if isinstance(i, dict)][:6]
        terms = [{**i, "kind": "term", "answer": i.get("term")} for i in terms]
        items = list(enumerate(facts)) + [(6 + k, i) for k, i in enumerate(terms)]
    else:
        items = list(enumerate(i for i in out.get("items", []) if isinstance(i, dict)))
    rows = []
    for k, it in items:
        kind = KIND.get(str(it.get("kind", "")).lower())
        q, a = str(it.get("question") or "").strip(), str(it.get("answer") or "").strip()
        if not kind or not a:
            continue
        r = {
            "qid": f"{c['chunk_id']}:{'f' if forced else 'q'}{k}",
            "chunk_id": c["chunk_id"],
            "kind": kind,
            "question": q or None,
            "answer": a,
            "term": None,
            "definition": None,
            "what": str(it.get("what") or "").strip() or None,
        }
        if kind == "term" and not forced:  # the term is the "question", the definition the answer
            r.update(question=None, answer=q, term=q, definition=a)
        elif kind == "term":
            r.update(term=a, definition=str(it.get("definition") or "").strip() or None)
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=30)
    args = ap.parse_args()
    set_rpm(args.rpm)
    pool = read_jsonl(WORK / "pool.jsonl")
    print(f"{len(pool)} chunks -> {TEACHER}")
    out = [q for qs in pmap(questions, pool) for q in qs]
    write_jsonl(WORK / "questions.jsonl", out)
    origin = {c["chunk_id"]: c["origin"] for c in pool}
    stats = {
        "chunks": len(pool),
        "chunks_with_items": len({q["chunk_id"] for q in out}),
        "items": dict(Counter(f"{origin[q['chunk_id']]}:{q['kind']}" for q in out)),
    }
    update_stats("questions", stats)
    print(stats)
    report_failures()


if __name__ == "__main__":
    main()
