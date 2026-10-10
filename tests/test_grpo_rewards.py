"""Stage 5's reward (train/grpo_rewards.py) on hand-written completions, before any sampling.

Each fixture is (completion text, expected format, expected correct). The closed-book correctness
rule itself is eval/scorers.qa_strict, with its own fixtures in tests/test_qa_strict.py; here the
format gate, the one-candidate rule, termination, the citation and abstain rules, the length
penalty and the TRL-facing functions. Fixtures are test inputs written for this file, never
training data (rule 13).
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
import grpo_rewards as R

EOS = [17, R.EOS]  # a terminated completion's ids (content token, </s>)
CUT = [17, 18]  # truncated: no </s>
IDS = ["fema-p-751:p462:c0", "fhwa-hif22031:p55:c0", "nist-gcr-13-917-24:p15:c0", "usace-em:p3:c1"]
GOLD = IDS[1]

CLOSED_VALUE = {"type": "closed_book", "gold": "10 ft", "kind": "value", "tolerance": 0.02}
CLOSED_IDENT = {
    "type": "closed_book",
    "gold": "Section 17.8.2",
    "kind": "identifier",
    "tolerance": 0.02,
}
CLOSED_TERM = {"type": "closed_book", "gold": "cripple wall", "kind": "term", "tolerance": 0.02}
CITE = {"type": "cite", "gold_chunk": GOLD, "chunk_ids": IDS, "max_distinct": 2}
ABSTAIN = {"type": "abstain", "phrase": R.ABSTAIN, "chunk_ids": IDS}

CASES = [  # (verifier, text, ids, format, correct)
    # closed-book: one line, one candidate, terminated
    (CLOSED_VALUE, "10 ft", EOS, True, True),
    (CLOSED_VALUE, "10 feet", EOS, True, True),
    (CLOSED_VALUE, "Answer: 10 ft", EOS, True, True),  # a label is stripped, as in the eval
    (CLOSED_VALUE, "12 ft", EOS, True, False),  # wrong value: format passes, correctness fails
    (CLOSED_VALUE, "10 m", EOS, True, False),  # right number, wrong unit
    (CLOSED_VALUE, "10 to 15 ft", EOS, False, False),  # a range: two values on the line
    (CLOSED_VALUE, "10 ft or 12 ft", EOS, False, False),  # two candidates
    (CLOSED_VALUE, "10 ft\nActually, 12 ft.", EOS, False, False),  # answer then a contradiction
    (CLOSED_VALUE, "10 ft\n10 ft", EOS, False, False),  # two lines
    (CLOSED_VALUE, "10 ft", CUT, False, False),  # truncated: never correct
    (CLOSED_VALUE, "", EOS, False, False),
    (CLOSED_IDENT, "Section 17.8.2", EOS, True, True),
    (CLOSED_IDENT, "17.8.2", EOS, True, True),  # locator word stripped
    (CLOSED_IDENT, "ASCE/SEI 7-22 Section 17.8.2", EOS, True, True),  # names the document
    (CLOSED_IDENT, "Section", EOS, True, False),  # a piece of the gold
    (CLOSED_IDENT, "Section 17.8.2.3", EOS, True, False),  # a child section
    (CLOSED_IDENT, "Section 17.8.2 or 17.8.3", EOS, False, False),
    (CLOSED_TERM, "cripple walls", EOS, True, True),
    (CLOSED_TERM, "wall", EOS, True, False),
    (CLOSED_TERM, "cripple wall; shear wall", EOS, False, False),
    (CLOSED_TERM, "cripple wall, shear wall", EOS, False, False),  # a comma list: a format failure
    (CLOSED_TERM, "shear wall, cripple wall", EOS, False, False),
    (CLOSED_IDENT, "17.8.3, Section 17.8.2", EOS, False, False),
    # grounded: valid citations; correct = cites the gold passage, at most 2 passages, no abstain
    (CITE, f"The width is 2 in. [{GOLD}]", EOS, True, True),
    (CITE, f"The width is 2 in. [{GOLD}] It is temporary [{IDS[0]}]", EOS, True, True),
    (CITE, f"Width 2 in. [{GOLD}] [{IDS[0]}] [{IDS[2]}] [{IDS[3]}]", EOS, True, False),  # cite all
    (CITE, f"Width 2 in. [{IDS[0]}]", EOS, True, False),  # a non-gold passage only
    (CITE, f"Width 2 in. [{GOLD}] [sic]", EOS, False, False),  # a bracket that isn't an id
    (CITE, "Width 2 in. [doc:p12:c0]", EOS, False, False),  # the prompt's format example id
    (CITE, "The width is 2 in.", EOS, False, False),  # no citation
    (CITE, R.ABSTAIN, EOS, False, False),  # a false abstain
    (CITE, f"{R.ABSTAIN} [{GOLD}]", EOS, True, False),  # cites, but abstains
    (CITE, f"The width is 2 in. [{GOLD}]", CUT, False, False),
    # abstain: exactly the sentence
    (ABSTAIN, "Not in the provided passages.", EOS, True, True),
    (ABSTAIN, "not in the provided passages", EOS, True, True),
    (ABSTAIN, "Not in the provided passages. The value is 5 ft.", EOS, False, False),
    (ABSTAIN, f"The value is 5 ft [{IDS[0]}]", EOS, True, False),  # answered, in grounded form
    (ABSTAIN, "The value is 5 ft.", EOS, False, False),
    (ABSTAIN, "Not in the provided passages.", CUT, False, False),
]


@pytest.mark.parametrize("v,text,ids,fmt,correct", CASES)
def test_score(v, text, ids, fmt, correct):
    s = R.score(text, ids, v)
    assert (s["format"], s["correct"]) == (fmt, correct), s


def test_totals_and_gate():
    right = R.score("10 ft", EOS, CLOSED_VALUE)
    wrong = R.score("12 ft", EOS, CLOSED_VALUE)
    hedge = R.score("10 to 15 ft", EOS, CLOSED_VALUE)
    assert R.total(right) == pytest.approx(1.0)
    assert R.total(wrong) == pytest.approx(0.1)
    assert R.total(hedge) == 0.0  # the gate: correctness is not even read
    assert R.total(right) > R.total(wrong) > R.total(hedge)


def test_length_penalty():
    free, cap = R.MAX_COMPLETION - 64, R.MAX_COMPLETION
    pen = lambda n: R.score("x", [17] * (n - 1) + [R.EOS], CLOSED_VALUE)["length"]
    assert pen(free) == 0.0
    assert pen(free + 32) == pytest.approx(-0.05)
    assert pen(cap) == pytest.approx(-0.1)
    assert R.score("x", [17] * cap, CLOSED_VALUE)["length"] == pytest.approx(-0.1)  # truncated
    # a truncated completion is strictly the worst: 0 + 0 - 0.1
    assert R.total(R.score("10 ft", [17] * cap, CLOSED_VALUE)) < R.total(wrong_short())


def wrong_short():
    return R.score("12 ft", EOS, CLOSED_VALUE)


def test_trl_functions():
    """TRL calls each function with the batch: conversational completions, the sampled ids and the
    dataset's verifier column (a JSON string)."""
    comps = [[{"role": "assistant", "content": t}] for t in ("10 ft", "12 ft", "10 to 15 ft")]
    ids = [EOS, EOS, EOS]
    ver = [json.dumps(CLOSED_VALUE)] * 3
    kw = {"prompts": [None] * 3, "trainer_state": None}
    assert R.format_reward(comps, ids, ver, **kw) == [0.1, 0.1, 0.0]
    assert R.correctness_reward(comps, ids, ver, **kw) == [0.9, 0.0, 0.0]
    assert R.length_penalty(comps, ids, ver, **kw) == [0.0, 0.0, 0.0]


def test_verifier_for():
    cb = R.verifier_for("closed_book", {"gold": "10 ft", "kind": "value"})
    assert cb == {"type": "closed_book", "gold": "10 ft", "kind": "value", "tolerance": 0.02}
    passages = [{"label": f"P{k + 1}", "chunk_id": c} for k, c in enumerate(IDS)]
    g = R.verifier_for("grounded", {"passages": passages, "gold_label": "P2"})
    assert g == {"type": "cite", "gold_chunk": GOLD, "chunk_ids": IDS, "max_distinct": 2}
    a = R.verifier_for("abstain", {"passages": passages})
    assert a == {"type": "abstain", "phrase": R.ABSTAIN, "chunk_ids": IDS}


@pytest.mark.skipif(
    not (REPO / "results/runs/sft-from-cpt/samples/dpo_pool.jsonl").exists(),
    reason="no pool samples",
)
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/sft/work/judged.jsonl").exists(),
    reason="data/sft/work/ is the SFT builder's gitignored output (make sft-data)",
)
def test_stage4_pool_characterisation():
    """The reward's closed-book verdicts on Stage 4's pool samples (sft-from-cpt, T 0.7, 4 per prompt),
    pinned: a change to the verifier shows here (1,299 correct; 1,305 before the 2026-10-09 comma
    fix, which fails one "triangular or trapezoidal cross sections", and the year fix, which fails
    five wrong years within 2% of "2010"; same_fact passed 1,492)."""
    sys.path.insert(0, str(REPO / "data/scripts"))
    gold = {}
    for line in (REPO / "data/sft/work/judged.jsonl").open():
        r = json.loads(line)
        gold[r["eid"]] = r
    correct = n = 0
    for line in (REPO / "results/runs/sft-from-cpt/samples/dpo_pool.jsonl").open():
        row = json.loads(line)
        if row["format"] != "closed_book":
            continue
        v = R.verifier_for("closed_book", gold[row["id"]])
        for s in row["samples"]:
            n += 1
            correct += R.score(s["text"], s["token_ids"], v)["correct"]
    assert n == 4336
    assert correct == 1299
