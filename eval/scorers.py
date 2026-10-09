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


# The strict closed-book checker (2026-10-09), written when Stage 4's verifier (sft_judge.same_fact)
# was about to become the GRPO reward: same_fact passed any piece of the gold ("Section" for
# "Section 17.8.2") and judged multi-number golds on their first number ("class 8" for "8 x 19"),
# and qa_correct above passes a range ("10 to 15 ft" for "10 ft"), a wrong unit ("800 ksi" for
# "800 psi") and a gold buried in any first line. Rules, by the answer's kind:
#   value (kind value/number, gold with a number): every gold number appears in the answer within
#     tolerance, in a compatible unit (the same after UNITS spellings, or either side unitless), and
#     no other answer number carries a gold number's unit (a conflicting value). Number words
#     zero-twelve read as digits; "2-1/2" is 2 and 0.5.
#   identifier with a number: designations normalised ("Article/Section/§" dropped, "C 1138" =
#     "C1138", "FEMA P-361" = "FEMA 361", an ASTM metric companion "/A820M" dropped), then every
#     gold word is in the answer and the answer's last numbered token is the gold's. A named
#     document passes ("AASHTO LRFD Article 6.5.4.2" for "6.5.4.2"); a fragment ("Section" for
#     "Section 17.8.2", "EM" for "EM 1110-2-2400"), a child or sibling section ("4.6.2.1.8" for
#     "4.6.2.1") and a dropped part of the gold ("Article C4.6.2.6.1" for "AASHTO LRFD Article
#     C4.6.2.6.1") fail. No alias lists exist (one gold string per item), so this is the rule-derived
#     stand-in for "full match against the alias list".
#   term (and any other gold without a number): one of the gold's forms (the gold, or without a
#     trailing parenthetical: "polar moment of inertia (Jr)") is the answer, or a whole phrase in it
#     with at most CONTEXT_WORDS more words ("Level III operations" for "Level III"); case,
#     punctuation, hyphens, articles and plural s ignored. Never a piece of the gold: "Collapse"
#     fails "low likelihood of collapse", "beam analogy" fails "flat-beam analogy".
#   a value gold's trailing parenthetical is a conversion and may be left out ("100 mm" for
#     "100 mm (4 in.)"); "8x19" = "8 x 19".
#   an "N:1" ratio gold is the value N ("10" for "10:1", the 2026-10-04 eval audit's ruling).
#   every kind: one candidate. "or", "and/or", "vs" or ";" the gold lacks, or (values) a range the
#     gold lacks, fails ("two values on the answer line"; the GRPO reward's format gate).
STRICT_KINDS = {
    "value": "value",
    "number": "value",
    "identifier": "phrase",
    "term": "phrase",
    "other": "phrase",
}
# sft_judge.NUMBER_WORDS
NUMBER_WORDS = {w: str(k) for k, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
)}  # fmt: skip
_QTY = re.compile(r"(?<![\w.,/])(-?\d[\d,]*(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?")
_UNIT = re.compile(r"\s*-?\s*([A-Za-z]+|[%°′″'\"])")
UNITS = {a: canon for canon, aliases in {
    "ft": ("ft", "feet", "foot", "'", "′"), "in": ("in", "inch", "inches", '"', "″"),
    "m": ("m", "meter", "meters", "metre", "metres"), "mm": ("mm", "millimeter", "millimeters"),
    "cm": ("cm", "centimeter", "centimeters"), "km": ("km", "kilometer", "kilometers"),
    "mi": ("mi", "mile", "miles"), "yd": ("yd", "yard", "yards"),
    "lb": ("lb", "lbs", "lbf", "pound", "pounds"), "kip": ("kip", "kips"), "kn": ("kn",),
    "kg": ("kg", "kilogram", "kilograms"), "psi": ("psi",), "ksi": ("ksi",), "psf": ("psf",),
    "pcf": ("pcf",), "pa": ("pa", "pascal", "pascals"), "kpa": ("kpa",), "mpa": ("mpa",), "gpa": ("gpa",),
    "%": ("%", "percent", "pct"), "s": ("s", "sec", "second", "seconds"),
    "min": ("min", "minute", "minutes"), "h": ("h", "hr", "hrs", "hour", "hours"), "day": ("day", "days"),
    "week": ("week", "weeks"), "month": ("month", "months"), "yr": ("yr", "yrs", "year", "years"),
    "deg": ("deg", "degree", "degrees", "°"), "mph": ("mph",), "hz": ("hz", "hertz"),
}.items() for a in aliases}  # fmt: skip
_ALT = re.compile(r"\b(?:or|and/or|either|versus|vs)\b|;", re.IGNORECASE)
# a range: a number, up to a unit's worth of text ("20% to 100%", "3” to 4”"), then to or a dash
_RANGE = re.compile(r"\d[^\d;]{0,12}?(?:\bto\b|[-–—])\s*[\d.]")
CONTEXT_WORDS = 3
_LOCATOR = re.compile(r"(?:\b(?:articles?|sections?)\b|§+)", re.IGNORECASE)


def _digits(t: str) -> str:
    words = "|".join(NUMBER_WORDS)
    return re.sub(rf"\b({words})\b", lambda m: NUMBER_WORDS[m[1].lower()], t, flags=re.IGNORECASE)


def quantities(text: str) -> list[tuple[float, str | None]]:
    """Every number in text with the unit written right after it (canonical UNITS name or None)."""
    out, d = [], re.sub(r"(\d)\s*[x×]\s*(?=\d)", r"\1 x ", _digits(text))
    for m in _QTY.finditer(d):
        try:
            v = float(m[1].replace(",", ""))
            if m[2]:
                v /= float(m[2])
        except (ValueError, ZeroDivisionError):
            continue
        u = _UNIT.match(d, m.end())
        out.append((v, UNITS.get(u[1].lower()) if u else None))
    return out


def _close(a: float, g: float, tol: float) -> bool:
    return abs(a) < 1e-9 if g == 0 else abs(a - g) / abs(g) <= tol + 1e-12


def _norm_id(text: str) -> str:
    t = _LOCATOR.sub(" ", text.replace("’", "'").replace("‘", "'"))
    t = re.sub(r"/\s*[A-Za-z]?\d+M\b", "", t)  # ASTM's metric companion: "A820/A820M"
    t = re.sub(r"\b([A-Za-z]) (?=\d)", r"\1", t)  # "C 1138" = "C1138" (before normalize drops "a")
    t = re.sub(
        r"(?<=[A-Za-z] )([A-Z])\b(?![-'’\d])", r"\1_", t
    )  # "Appendix A": a designation, not an article
    t = normalize(t)
    t = re.sub(r"\bfema p ?(?=\d)", "fema ", t)  # "FEMA P-361" = "FEMA 361"
    return re.sub(r"(?<=[a-z]{3})s\b", "", t).rstrip(". ")


def _forms(text: str) -> set[str]:
    return {f for t in (text, re.sub(r"\s*\([^()]*\)\s*$", "", text)) if (f := _norm_id(t))}


def _term_ok(line: str, gold: str) -> bool:
    a = _forms(line)
    for g in _forms(gold):
        if g in a or any(
            re.search(rf"(?<!\w){re.escape(g)}(?!\w)", f)
            and len(f.split()) <= len(g.split()) + CONTEXT_WORDS
            for f in a
        ):
            return True
    return False


def _identifier_ok(line: str, gold: str) -> bool:
    a, g = _norm_id(line).split(), _norm_id(gold).split()
    numbered = lambda ws: [w for w in ws if re.search(r"\d", w)]
    return set(g) <= set(a) and bool(numbered(a)) and numbered(a)[-1] == numbered(g)[-1]


def one_candidate(line: str, gold: str, kind: str) -> bool:
    if _ALT.search(line) and not _ALT.search(gold):
        return False
    rng = lambda t: _RANGE.search(_digits(t))
    return not (STRICT_KINDS.get(kind) == "value" and rng(line) and not rng(gold))


def qa_strict_reason(pred: str, gold: str, kind: str, tolerance: float = 0.02) -> str | None:
    """None if the answer (first line of pred) is right under the strict rules, else why not:
    two_candidates, missing_number, unit, conflict or phrase."""
    line = first_line(pred)
    if not line:
        return "phrase"
    if not one_candidate(line, gold, kind):
        return "two_candidates"
    ratio = _RATIO_TO_ONE.match(gold)
    if ratio:
        gold, line = ratio[1], re.sub(r"\s*:\s*1\b", "", line)
    gq = quantities(gold) if STRICT_KINDS.get(kind, "phrase") == "value" or ratio else []
    if not gq:
        if kind == "identifier" and re.search(r"\d", gold):
            return None if _identifier_ok(line, gold) else "phrase"
        return None if _term_ok(line, gold) else "phrase"
    bare = re.sub(r"\s*\([^()]*\)\s*$", "", gold)
    if bare != gold and quantities(bare) and _values(line, quantities(bare), tolerance) is None:
        return None
    return _values(line, gq, tolerance)


def _values(line: str, gq: list, tolerance: float) -> str | None:
    aq, used = quantities(line), set()
    compat = lambda u, w: u is None or w is None or u == w
    for gv, gu in gq:
        hit = next((i for i, (av, au) in enumerate(aq)
                    if i not in used and _close(av, gv, tolerance) and compat(au, gu)), None)  # fmt: skip
        if hit is None:
            return "unit" if any(_close(av, gv, tolerance) for av, _ in aq) else "missing_number"
        used.add(hit)
    gunits = {gu for _, gu in gq}
    if any(i not in used and au in gunits for i, (_, au) in enumerate(aq)):
        return "conflict"
    return None


def qa_strict(pred: str, gold: str, kind: str, tolerance: float = 0.02) -> bool:
    return qa_strict_reason(pred, gold, kind, tolerance) is None


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
