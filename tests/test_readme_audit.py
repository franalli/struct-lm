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
