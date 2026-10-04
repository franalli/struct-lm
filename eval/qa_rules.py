"""Rules on domain_qa items shared by the task builder (make_tasks.py, --task-version 2) and the
scorer side (run_eval.py's per-kind columns). Pure string handling, no model calls.

is_locator   rejects layout locators: answers that point into a document's own layout (page,
             table, figure, equation, exhibit or plate numbers, page counts) and questions that ask
             where in a document something appears. Layout, not knowledge, and it changes between
             editions. Document ids and article/section numbers are not locators: they are
             identifiers, kept and capped.
answer_kind  number | identifier | term | other, so Stage 3 can report which kinds of answer move.
             Identifiers (document ids, article/section numbers) are the hardest closed-book recall
             and the least useful to a user, so make_tasks caps them at 20% of the set.
"""

import re

LOCATOR_ANSWER = re.compile(
    r"^\s*(p\.|pp\.?|pages?|table|tbl\.?|figure|fig\.?|eq\.?|equation|exhibit|plate)"
    r"\s*[A-Z]{0,2}-?\d[\w.\-]*\s*$"
    r"|^\s*\d[\d,]*\s*(pp\.?|pages)\s*$",  # a page count
    re.IGNORECASE,
)
LOCATOR_ASK = re.compile(
    r"\b(which|what|on what|on which)\s+(page|table|figure|fig\.?|exhibit)\b"
    r"|\bwhere in (the|this) (document|manual|report|chapter)\b"
    r"|\bpage (number|count)\b|\bhow many pages\b",
    re.IGNORECASE,
)
# "which equation" asks for a locator only when the answer is a number ("Equation B-28"); a named
# equation ("Manning's equation") is a method, and a fair question.
EQUATION_ASK = re.compile(r"\b(which|what)\s+equation\b", re.IGNORECASE)

IDENTIFIER = re.compile(
    r"^(EM|ETL|ER|EP|FEMA\s*P?-?|NIST\s*(GCR|TN|SP|NCSTAR)?|NASA-STD|FHWA-|UFC|AASHTO|ACI|AISC|"
    r"ASCE|ASTM|AWS|ANSI|IBC|IRC)\s*[\w\-./]*\d"
    r"|^(AASHTO\s+LRFD\s+)?(article|art\.|section|sec\.|clause|chapter|appendix|§)\s*[A-Z]?\d[\w.\-]*$"
    r"|^\d+(\.\d+){2,}[a-z]?$",  # a bare clause number, 5.4.2.3.1 (two dots or more: 1.25 is a value)
    re.IGNORECASE,
)


def is_locator(question: str, answer: str) -> str | None:
    """The reason an item is a layout locator, or None."""
    if LOCATOR_ANSWER.match(answer):
        return "locator_answer"
    if LOCATOR_ASK.search(question):
        return "locator_ask"
    if EQUATION_ASK.search(question) and re.search(r"\d", answer):
        return "locator_equation_number"
    return None


def answer_kind(answer: str, answer_type: str) -> str:
    """number | identifier | term | other. Identifier first: "FEMA 361" has one number, so it is
    scored numerically, but it names a document."""
    a = answer.strip()
    if IDENTIFIER.match(a):
        return "identifier"
    if answer_type == "numeric" or re.match(r"^\d+(\.\d+)?\s*:\s*1$", a):  # 10:1 is a value
        return "number"
    if re.search(r"\d", a):
        return "other"  # ratios, ranges, mixed strings
    return "term" if len(a.split()) <= 5 else "other"
