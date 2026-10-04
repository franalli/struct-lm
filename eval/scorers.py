"""Deterministic scorers for the KPI tasks. No model calls in this file.

Used by run_eval.score():
  domain_qa    qa_correct()       exact_match or numeric_match, depending on the item's answer_type
  grounded     citations_valid()  format check only; whether the cited passage actually supports
                                  the answer is the judge's job (judge.grounded_rubric)
  adversarial  abstained()        fast path for the exact abstain phrase; anything else goes to
                                  the judge (judge.adversarial_rubric)
  vocab        (none)             judged only (judge.vocab_rubric)

Everything here is pure string handling, so re-scoring saved generations with
`run_eval.py --rescore` gives identical numbers on any machine.
"""

import re

# First number in a string: optional sign, digits with thousands commas ("1,200"), an optional
# decimal part, and optional scientific notation ("2.9e4"). Units and trailing text are ignored.
# A hyphen glued to a letter or digit is part of an id, not a minus sign: "FEMA P-361" is 361
# (it read as -361, so a correct "FEMA P-361" failed against a gold "FEMA 361").
_NUM = re.compile(r"(?:(?<![A-Za-z0-9])-)?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
# English articles, dropped so "the plastic moment" matches "plastic moment".
_ARTICLES = re.compile(r"\b(the|a|an)\b")
# Everything except word characters, whitespace and the symbols that carry meaning in engineering
# answers: "." (decimals, article numbers like 5.10.9), "%" (percentages), "/" (ratios, "L/360",
# "AASHTO/AWS"), "-" (already turned into spaces by normalize(), kept here for safety).
_PUNCT = re.compile(r"[^\w\s.%/-]")


# A passage-id citation such as [fhwa-nhi-15-047:p30:c0] (or a copied format example, [doc:p3:c0]).
_CHUNK_CITE = re.compile(r"\[[^\[\]]*:p\d+:c\d+[^\[\]]*\]")
# A leading "Definition:" / "Answer:" label, with any markdown around it.
_LABEL = re.compile(r"^\W*(definition|answer)\s*:\W*", re.IGNORECASE)


def substance(text: str) -> str:
    """The answer with passage-id citations and markdown emphasis removed. Empty means the model
    said nothing: run_eval scores that by rule, because the judge, shown an empty or
    citation-only answer, sometimes graded it as correct or as a refusal."""
    return _CHUNK_CITE.sub("", text).replace("*", "").replace("#", "").strip()


def answer_line(text: str) -> str:
    """First line that carries an answer, for chat models on the one-line tasks. Skips an echoed
    "Term: x" header and short intros ending in a colon ("Here's a precise definition:"), and
    strips a leading "Definition:" label and surrounding markdown. "" if no line qualifies."""
    for line in text.strip().splitlines():
        s = line.strip()
        if (
            not s
            or re.match(r"^\W*term\s*:", s, re.IGNORECASE)
            or (s.endswith(":") and len(s) < 80)
        ):
            continue
        return _LABEL.sub("", s).strip("* ").strip()
    return ""


def first_line(text: str) -> str:
    """The answer is the first non-empty line. Base models often continue past it (a new "Q:",
    an explanation); GEN stops at "\\n" for domain_qa, and this is the second guard."""
    return text.strip().split("\n", 1)[0].strip()


def normalize(text: str) -> str:
    """Canonical form for exact_match: first line, lowercased, hyphens as spaces,
    punctuation and articles removed, whitespace collapsed."""
    t = (
        first_line(text).lower().replace("-", " ")
    )  # "complete-joint-penetration" == "complete joint penetration"
    t = _PUNCT.sub(" ", t)
    t = _ARTICLES.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def exact_match(pred: str, gold: str) -> bool:
    """Normalised first line equals the gold answer, or contains it as a whole phrase.
    Containment is deliberate: base models often prefix 'the answer is'. It is only
    lenient on the first line, and max_tokens is small, so it cannot reward rambling.
    The lookarounds stop a gold "G" from matching inside "Grade 50"."""
    p, g = normalize(pred), normalize(gold)
    if not g:  # an empty gold answer would "match" every prediction
        return False
    return p == g or re.search(rf"(?<![\w]){re.escape(g)}(?![\w])", p) is not None


def _parse_number(text: str) -> float | None:
    """First number in text as a float, with thousands commas removed; None if there is none.
    The regex can match an unparseable fragment (e.g. "1.e"), hence the ValueError guard."""
    m = _NUM.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def numeric_match(pred: str, gold: str, rel_tol: float = 0.02) -> bool:
    """First number in the prediction's first line vs first number in gold, within a relative
    tolerance (2% by default, per item via "tolerance" in domain_qa.jsonl).

    Units are NOT compared: "800 psi" and "800 ksi" both match a gold "800 psi". This is
    accepted because questions name the quantity, and unit conversion would need a units
    library for little gain. make_tasks.answer_type_of() only marks answers with exactly one
    number as numeric, so "the first number" is well defined on the gold side."""
    p, g = _parse_number(first_line(pred)), _parse_number(gold)
    if p is None or g is None:
        return False
    if g == 0:  # relative tolerance is undefined at zero; require an exact zero instead
        return abs(p) < 1e-9
    return abs(p - g) / abs(g) <= rel_tol


# A ratio gold "10:1" (or "4 : 1"): two numbers, so make_tasks labels it exact, but its value is
# the first one. "aspect ratio 10" is a correct answer that exact match failed (qa-0542, audit
# 2026-10-04).
_RATIO_TO_ONE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*1\s*$")


def qa_correct(pred: str, gold: str, answer_type: str, tolerance: float = 0.02) -> bool:
    """domain_qa score for one item: numeric answers by value, everything else by text."""
    if answer_type == "numeric" or _RATIO_TO_ONE.match(gold):
        return numeric_match(pred, gold, tolerance)
    return exact_match(pred, gold)


# A bracketed span with no nested brackets: "[fhwa-nhi-15-047:p12:c0]". Non-greedy, so
# "[a] and [b]" gives two citations, not one.
_CITE = re.compile(r"\[([^\[\]]+?)\]")


def citations(text: str) -> list[str]:
    """Every bracketed citation in the answer, in order, whitespace-trimmed."""
    return [c.strip() for c in _CITE.findall(text)]


def citations_valid(text: str, context_ids: set[str]) -> bool:
    """At least one citation, and every citation is an id that was actually provided.

    Catches the two format failures: no citations at all, and invented ids (including the
    prompt's own example ids like "example-a:p3:c0", which aren't in context_ids). Says
    nothing about whether the cited passage supports the claim; that is cite_supported."""
    cites = citations(text)
    return bool(cites) and all(c in context_ids for c in cites)


def abstained(text: str, phrase: str) -> bool:
    """The output contains the exact abstain phrase (prompts.ABSTAIN_PHRASE), case-insensitive.
    Only a fast path: other ways of declining are left to the judge, so a False here is not
    a hallucination verdict on its own."""
    return phrase in text.lower()
