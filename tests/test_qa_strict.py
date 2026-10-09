"""The strict closed-book checker (eval/scorers.qa_strict), the GRPO reward's correctness rule.

Every known false pass of the Stage 4 verifier (sft_judge.same_fact) and of the eval's qa_correct
is a must-fail fixture; the cases the 2026-10-09 hand reads ruled correct are must-pass. Answers are
the answer line only (one candidate), golds as stored.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from scorers import qa_correct, qa_strict, qa_strict_reason

# (answer, gold, kind): same_fact passed every one (a piece of the gold, or the first number)
SAME_FACT_FALSE_PASSES = [
    ("Section", "Section 17.8.2", "identifier"),
    ("EM", "EM 1110-2-2400", "identifier"),
    ("2", "EM 1110-2-2400", "identifier"),
    ("wall", "cripple wall", "term"),
    ("1000 feet", "1000 meters", "value"),
    ("Section 11.4.2", "Section 11.4.2.1", "identifier"),
    ("class 8", "8 x 19", "value"),
    ("1 inch = 50 feet", "1 inch to 100 feet", "value"),
    ("Collapse", "low likelihood of collapse", "value"),
    ("Ordinary Method", "Ordinary Method of Slices", "term"),
    ("0.131″ x 2″", '0.131" x 2-1/2"', "value"),
    ("beam analogy", "flat-beam analogy", "term"),
    ("bronze", "self-lubricating bronze sleeves", "value"),
    ("C17.8.2", "C17.8.2.2", "identifier"),
    ("AISC 360", "AISC 360, Section I2.1b", "identifier"),
    ("Appendix E", "Appendix A", "identifier"),
    ("0.2 to 0.5", "0.2 to 0.7", "value"),
    ("3/16 inch", "3 inches", "value"),
    ("1 3/8 inches", "1 inch", "value"),
    ("Section 11.4.3", "ASCE/SEI 7-05 Section 11.4.3", "identifier"),  # drops part of the gold
    ("Article C4.6.2.6.1", "AASHTO LRFD Article C4.6.2.6.1", "identifier"),
]
# (answer, gold, kind): the eval's qa_correct passed every one (first number, no units, containment)
QA_CORRECT_FALSE_PASSES = [
    ("10 to 12 ft", "10 feet", "number"),
    ("12 to 18 inches", "12 in.", "number"),
    ("1.1 to 1.2", "1.1", "number"),
    ("1/2 inch", "1 in.", "number"),
    ("15-minute period", "15 days", "number"),
    ("800 ksi", "800 psi", "number"),
    ("4.6.2.1.8", "4.6.2.1", "identifier"),
    ("AASHTO LRFD Article 6.10.10.2", "6.10.10", "identifier"),
]
HEDGES = [  # one candidate per line: alternatives or a range the gold doesn't have
    ("Section 3.2 or 3.3", "Section 3.2", "identifier"),
    ("steel shape, angle, or tee", "steel shapes", "value"),
    ("3 to 5 ft", "3 ft", "value"),
    ("10 ft; 15 ft", "10 ft", "value"),
    # comma, slash and "and" lists of phrases (2026-10-09: found by reading the code), both orders
    ("cripple wall, shear wall", "cripple wall", "term"),
    ("shear wall, cripple wall", "cripple wall", "term"),
    ("cripple wall/shear wall", "cripple wall", "term"),
    ("cripple wall and shear wall", "cripple wall", "term"),
    ("17.8.3, Section 17.8.2", "Section 17.8.2", "identifier"),
    ("Section 17.8.2, 17.8.3", "Section 17.8.2", "identifier"),
    ("Section 17.8.2 / Section 17.8.3", "Section 17.8.2", "identifier"),
    ("Appendix A, Appendix B", "Appendix A", "identifier"),
    ("chamfers, fillets", "chamfers", "value"),
    # a gold holding a separator passes only exactly
    ("triangular or trapezoidal cross sections", "triangular or trapezoidal", "value"),
    ("1.05 to 1.1", "1.1", "value"),
    ("8–15 m", "8 m", "value"),
    ("1996", "2010", "value"),  # a year is exact: 2% of 2010 is 40 years (the Stage 5 hack audit)
]
MUST_PASS = [
    ("6.10.10", "Article 6.10.10", "identifier"),
    ("Article 6.10.10", "6.10.10", "identifier"),
    ("§ 6.10.10", "Article 6.10.10", "identifier"),
    ("AASHTO LRFD Article 6.5.4.2", "6.5.4.2", "identifier"),  # names the document
    ("ASCE/SEI 7-22 §12.11.2", "12.11.2", "identifier"),
    ("ASTM C1138", "ASTM C 1138", "identifier"),
    ("ASTM A820/A820M", "ASTM A 820", "identifier"),
    ("FEMA P-361", "FEMA 361", "identifier"),
    ("FEMA 750", "FEMA P-750", "identifier"),
    ("AISC Design Guide 33", "Design Guide 33", "identifier"),
    ("EM 1110-2-2400", "EM 1110-2-2400", "identifier"),
    ("five years", "5 years", "value"),
    ("three times", "3 times", "number"),
    ("1.0 inch = 25.4 mm", "1.0 in = 25.4 mm", "value"),
    ("0.0015 to 0.0020", "0.0015 to 0.002 in", "value"),
    ("40 percent", "40%", "number"),
    ("two stories", "2", "value"),
    ("10 ft (3 m)", "10 ft", "value"),
    ("100 mm", "100 mm (4 in.)", "value"),
    ("8x19", "8 x 19", "value"),
    ("10", "10:1", "number"),
    ("0.2 f’c", "0.2f’c", "number"),
    ("cripple walls", "cripple wall", "term"),
    ("polar moment of inertia", "polar moment of inertia (Jr)", "term"),
    ("Level III operations", "Level III", "identifier"),
    ("triangular or trapezoidal", "triangular or trapezoidal", "value"),
    ("Edris, Strohm, and Woo (1991)", "Edris, Strohm, and Woo (1991)", "term"),
    ("AISC 360, Section I2.1b", "AISC 360, Section I2.1b", "identifier"),
    ("1,000-year flood", "1000-year flood", "term"),  # a thousands comma is not a separator
    ("lowest practical w/c ratio", "lowest practical w/c", "value"),  # a ratio, not a list
    ("10 ft, 3 m", "10 ft", "value"),
    ("2010", "2010", "value"),  # numbers keep their rule: a conversion is not a candidate
    ("20 to 100 percent", "20% to 100%", "value"),  # the gold's own range, with units between
    ("3 to 4 inches", "3” to 4”", "value"),
]


@pytest.mark.parametrize(
    "answer,gold,kind", SAME_FACT_FALSE_PASSES + QA_CORRECT_FALSE_PASSES + HEDGES
)
def test_must_fail(answer, gold, kind):
    assert not qa_strict(answer, gold, kind), qa_strict_reason(answer, gold, kind)


@pytest.mark.parametrize("answer,gold,kind", MUST_PASS)
def test_must_pass(answer, gold, kind):
    assert qa_strict(answer, gold, kind), qa_strict_reason(answer, gold, kind)


def test_reasons():
    assert qa_strict_reason("10 to 15 ft", "10 ft", "value") == "two_candidates"
    assert qa_strict_reason("1000 feet", "1000 meters", "value") == "unit"
    assert qa_strict_reason("class 8", "8 x 19", "value") == "missing_number"
    assert qa_strict_reason("10 ft, not 15 ft", "10 ft", "value") == "conflict"
    assert qa_strict_reason("Section", "Section 17.8.2", "identifier") == "phrase"
    assert qa_strict_reason("", "10 ft", "value") == "phrase"


def test_first_line_only():
    assert not qa_strict("bottom\nof the heat exchanger", "bottom of the heat exchanger", "value")


@pytest.mark.parametrize("answer,gold,kind", QA_CORRECT_FALSE_PASSES)
def test_qa_correct_was_lenient(answer, gold, kind):
    """The eval's scorer passes these (why table.md now carries qa_strict next to qa_acc)."""
    numeric = len(__import__("re").findall(r"\d[\d,]*(?:\.\d+)?", gold)) == 1
    assert qa_correct(answer, gold, "numeric" if numeric else "exact")
