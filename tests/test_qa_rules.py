"""eval/qa_rules.py: the layout-locator filter and the answer-kind tags."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from qa_rules import answer_kind, is_locator

LOCATORS = [  # (question, answer): real candidates from the 2026-10-04 generation
    ("Which ASCE/SEI 7-16 table defines an out-of-plane offset irregularity?", "Table 12.3-1"),
    ("Which AASHTO LRFD equation gives Mn for a non-composite box section?", "Eq. 6.12.2.2.2-1"),
    (
        "In USACE EM 1110-2-2104, what equation determines the balanced eccentricity ratio?",
        "Equation B-28",
    ),
    ("Which ASCE 7-16 figure provides the deterministic MCER lower limit?", "Figure 21.2-1"),
    ("Which reference plate in USACE EM 1110-2-1601 gives hydraulic jump lengths?", "Plate 51"),
    ("What is the page count of FEMA 232?", "212 pages"),
    ("What is the page count of Bleich's 1952 book on buckling?", "508 pp"),
    ("How many pages does FHWA-HIF-19-094 contain?", "65"),
    ("What is the page number of the fiber beam-column formulation in EESD?", "711"),
    ("Where in the manual is the cofferdam design procedure described?", "Chapter 4"),
]
NOT_LOCATORS = [
    ("According to Table 3.4.1-1, what is the load factor for live load at Strength I?", "1.75"),
    (
        "In USACE EM 1110-2-1417, what equation is used for normal depth in hydrologic routing?",
        "Manning's equation",
    ),
    (
        "What plate buckling coefficient applies to outstanding legs of single angles in AASHTO LRFD?",
        "0.45",
    ),
    ("Which NIST standard governs performance of wood-based structural-use panels?", "PS2-10"),
    ("Which USACE EM provides detailed design guidance for pile foundations?", "EM 1110-2-2906"),
    ("Which AASHTO LRFD article governs shear resistance of an unstiffened web?", "6.10.9.2"),
    (
        "What is the minimum mortar shear strength for dynamic stability of cracked URM walls?",
        "30 lb/in²",
    ),
    ("What model predicts sea-level-rise effects on sandy coasts in EM 1110-2-1810?", "Bruun rule"),
    (
        "What is the maximum aspect ratio for grid elements in a groundwater model per EM 1110-2-1421?",
        "10:1",
    ),
    ("Per FEMA P-751 Table 12.2-1, what response modification coefficient applies to SMFs?", "8"),
]


@pytest.mark.parametrize(("question", "answer"), LOCATORS)
def test_locators_rejected(question, answer):
    assert is_locator(question, answer)


@pytest.mark.parametrize(("question", "answer"), NOT_LOCATORS)
def test_values_and_identifiers_kept(question, answer):
    assert is_locator(question, answer) is None


@pytest.mark.parametrize(
    ("answer", "answer_type", "kind"),
    [
        ("0.75", "numeric", "number"),
        ("24 ksi", "numeric", "number"),
        ("10:1", "exact", "number"),
        ("1.25", "numeric", "number"),
        ("EM 1110-2-2906", "exact", "identifier"),
        ("FEMA 361", "numeric", "identifier"),
        ("FEMA P-695", "numeric", "identifier"),
        ("6.10.9.2", "exact", "identifier"),
        ("Article 5.4.2.6", "exact", "identifier"),
        ("Bruun rule", "exact", "term"),
        ("20 to 30 years", "exact", "other"),
    ],
)
def test_answer_kind(answer, answer_type, kind):
    assert answer_kind(answer, answer_type) == kind
