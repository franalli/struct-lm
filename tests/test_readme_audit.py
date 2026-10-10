"""train/readme_audit.py: every number in the README's prose traces to a generated table or a named
file, the prose stays under 3,000 words, and the lifecycle diagram names only real rows."""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

import readme_audit


def test_readme_numbers_trace():
    problems = readme_audit.audit()
    assert not problems, "\n".join(problems)


def test_readme_word_budget():
    assert readme_audit.word_count() < 3000


def test_audit_catches_untraced_and_misattributed(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(
        "Intro.\n\nThe gap is 0.41 nats, 28.7% on seen facts, in Stage 3 on 2026-10-09.\n\n"
        "<!-- headline-summary:start -->\n| x | 28.7 |\n<!-- headline-summary:end -->\n"
    )
    side = tmp_path / "numbers.tsv"
    side.write_text("number\tcontext\tsource\n")
    monkeypatch.setattr(readme_audit, "SIDECAR", side)
    problems = readme_audit.audit(readme)
    # 28.7 has three significant digits and sits in a generated block; 0.41 needs the sidecar;
    # the stage number and the date are not numbers to trace
    assert len(problems) == 1 and ": 0.41 " in problems[0]
    side.write_text("number\tcontext\tsource\n0.41\tgap is 0.41\tresults/train_runs.md\n")
    assert readme_audit.audit(readme) == []
    side.write_text("number\tcontext\tsource\n0.41\tgap is 0.41\tpyproject.toml\n")
    assert any("not in pyproject.toml" in p for p in readme_audit.audit(readme))


def test_lifecycle_names_real_rows():
    """The README's lifecycle diagram (docs/diagrams/src/lifecycle.json, rendered to
    docs/diagrams/lifecycle.svg) names only runs that are rows of results/table.md."""
    import json

    assert "docs/diagrams/lifecycle.svg" in (REPO / "README.md").read_text()
    ir = json.loads((REPO / "docs/diagrams/src/lifecycle.json").read_text())
    labels = " ".join(n["label"] for n in ir["nodes"])
    table = (REPO / "results/table.md").read_text()
    rows = set(re.findall(r"^\| ([a-z0-9.-]+) \|", table, re.MULTILINE))
    names = set(
        re.findall(r"\b(base-8b-hf|cpt-8b[\w-]*|sft-from-[\w-]+|dpo[\w-]*|grpo[\w-]*)", labels)
    )
    assert names and names <= rows, names - rows


def test_diagram_numbers_are_checked(tmp_path):
    """A number in a stage diagram's IR needs a sidecar row scoped to that file (`in`)."""
    import json

    ir = tmp_path / "stage9.json"
    ir.write_text(
        json.dumps({"nodes": [{"id": "a", "label": "463 strict pairs\n(445 train)"}], "edges": []})
    )
    lines = readme_audit.diagram_lines(ir)
    assert lines == [(1, "463 strict pairs (445 train)")]
    problems = readme_audit.check("stage9.json", lines, [], [])
    assert {p.split(": ")[1].split()[0] for p in problems} == {"463", "445"}
    rows = [
        {"number": "463", "context": "463 strict", "source": "x"},
        {"number": "445", "context": "445 train", "source": "x"},
    ]
    assert readme_audit.check("stage9.json", lines, [], rows) == []


# Disclosures earlier reviews added on purpose; an edit for length must not drop them.
CAVEATS = [
    "CPT missed its pre-registered target",  # its registered perplexity target failed
    "hallucination is worse than Instruct's",  # SFT against Instruct, at the seed gap
    "two-arm floor added after the analysis",  # DPO's hallucination metric depends on the floor
    "in a metric added after the fact",  # DPO's pass@1 metric was post hoc
    "the mean of its two seeds",  # GRPO's figures are two-seed means
    "1 df per arm",  # two seeds per arm
    "grew from 130 to 322 items after Stage 2",  # the eval changed after a read
    "includes 0",  # the unseen SFT-vs-Instruct CI
    "the untrained base also scores",  # the unseen lead is the base's too
    "the highest open-loop rate measured",  # 16 req/s is not a measured limit
    "Single-seed rows are read against the same floor",
    "rewritten after it fired",  # the entropy stop rule was post hoc
    "only after Stage 4's analysis",  # the two-arm floor change was post hoc
    "not on merit",  # why dpo-strict, not sft-from-cpt, is served
    "Retrieval is untested",  # the grounded eval supplies the gold passage
    "Chat behaviour beyond the eval is untested",  # no instruction-following or safety evals
    "No structural engineer saw items or outputs",  # no domain-expert review
    "No minimum detectable effect was set",  # the eval was not sized for 3-point gains
    "are fixtures and are never trained on",  # rule 13
]


def test_readme_keeps_its_caveats():
    text = " ".join((REPO / "README.md").read_text().split())  # line wrapping doesn't matter
    missing = [c for c in CAVEATS if c not in text]
    assert not missing, missing
