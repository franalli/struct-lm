"""data/sft/hosted/: the committed SFT set, with the Tülu 3 replay records' text blanked and their
row ids kept (data/scripts/sft_replay_fetch.py rebuilds data/sft/{train,sft_val}.jsonl from them
and checks SHA256SUMS). No third-party replay text is hosted; every other record is committed
whole, and matches the built file wherever one exists."""

import hashlib
import json
from pathlib import Path

import pytest

SFT = Path(__file__).resolve().parents[1] / "data/sft"
FILES = ("train.jsonl", "sft_val.jsonl")


def digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("name", FILES)
def test_replay_text_not_hosted(name):
    hosted = rows(SFT / "hosted" / name)
    replay = [r for r in hosted if r["format"] == "replay"]
    assert replay, name
    for r in replay:
        assert r["prompt"] is None and r["completion"] is None, r["eid"]
        if r["eid"].startswith("sha256:"):  # withdrawn: only digests of its ids are kept
            assert r["replay_source"].startswith("sha256:")
        else:
            assert r["eid"].startswith("replay:") and r["replay_source"].endswith(
                "#" + r["eid"][7:]
            )
    assert all(r["prompt"] and r["completion"] for r in hosted if r["format"] != "replay")


@pytest.mark.parametrize("name", FILES)
def test_hosted_matches_built(name):
    built = SFT / name
    if not built.exists():
        pytest.skip(f"data/sft/{name} not rebuilt (data/scripts/sft_replay_fetch.py)")
    for h, b in zip(rows(SFT / "hosted" / name), rows(built), strict=True):
        if b["format"] == "replay":
            b = {**b, "prompt": None, "completion": None}
            if h["eid"].startswith("sha256:"):
                b = {**b, "replay_source": digest(b["replay_source"]), "eid": digest(b["eid"])}
        assert h == b
