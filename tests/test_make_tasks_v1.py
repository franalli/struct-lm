"""make_tasks.py --task-version 1 still rebuilds the eval frozen on 2026-09-27 byte for byte (the
130-item domain_qa behind results/table_v1.md), --task-version 2 (the default) rebuilds the v2 set
of 2026-10-04 (results/table_v2.md) byte for byte, and --task-version 3 rebuilds the committed
eval/tasks/ (current since Stage 3), which only removes items from v2 (the few-shot passages).
Runs from the LLM disk cache, so it makes no API calls when the cache is
complete; skipped where the cache, chunks or key aren't available."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "eval/tasks"
FILES = ("domain_qa", "grounded", "vocab", "adversarial", "fewshot")
# sha256 of the v1 files as committed before the v2 build (git show 51cc909:eval/tasks/<file>)
V1 = {
    "domain_qa.jsonl": "6270f0eee7300abfcf0e6ad7061303804a5beb3b56c8fa257d312ce63a96ba51",
    "grounded.jsonl": "5fbf508485a318a78d62d2925f36cd27d7e6327d37c1dc022c7419c328450ffd",
    "vocab.jsonl": "76db88bf1f15d5979d12fb70353fea67c74e1a56622c117f8eeb62ad9e98ee91",
    "adversarial.jsonl": "bc934a939bc41b1e5a712bcb53421e7a6185dae1a9933e70c23de1d741d2bdf2",
    "fewshot.jsonl": "b513522a835282207cde65973a21ea3d491d0398ad67028782b70cd43bf33812",
    "eval_chunk_ids.txt": "9de59d11abd216ee85e0b1225a06841dfb03a4475cea386651f1bec6362af34d",
}
# sha256 of the v2 files as committed before v3 replaced them (git show 2bd5bf3:eval/tasks/<file>)
V2 = {
    "domain_qa.jsonl": "6e73e19bd0918e0a2d237e9c803b42d86f19fa54446ab86f4c0b770a0373ae86",
    "held_back.jsonl": "4575441007522607832f321bba5edc088040a71935e5ad83102b222b98e04ebf",
    "locators.jsonl": "bffcf986346d245d34020ec01cd0ea902987d78e0458bf2cb949b53b76c2a44e",
    **{k: v for k, v in V1.items() if k not in ("domain_qa.jsonl", "eval_chunk_ids.txt")},
    "eval_chunk_ids.txt": "2e46626111da76c920c338b1906656f1e2b8b02a87329ab74d91969c6952174d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


NEEDS = pytest.mark.skipif(
    not (REPO / "eval/.cache/llm_cache.jsonl").exists()
    or not (REPO / "data/processed/chunks.jsonl").exists()
    or not os.environ.get("MISTRAL_API_KEY"),
    reason="needs the LLM cache, data/processed/chunks.jsonl and MISTRAL_API_KEY",
)


def build(out: Path, version: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("rejects.jsonl", "answer_kinds.jsonl"):
        if (FROZEN / name).exists():
            shutil.copy(FROZEN / name, out)
    subprocess.run(
        [sys.executable, "eval/make_tasks.py", "--out", str(out), "--task-version", str(version)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )


@NEEDS
def test_task_version_1_reproduces_frozen_set(tmp_path):
    build(tmp_path, 1)
    for name, digest in V1.items():
        assert sha(tmp_path / name) == digest, name


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@NEEDS
def test_task_version_3_only_drops_shot_passages(tmp_path):
    """v3 = v2 with the few-shot split off by passage: removal only, so every v3 item is a reviewed
    v2 item under the same id, and none comes from a shot's passage."""
    build(tmp_path / "v2", 2)
    v2 = {d["id"]: d for d in load(tmp_path / "v2/domain_qa.jsonl")}
    v3 = load(FROZEN / "domain_qa.jsonl")
    shots = {s["source_chunk"] for s in load(FROZEN / "fewshot.jsonl")}
    assert not [d["id"] for d in v3 if d["source_chunk"] in shots]
    assert all(d == v2[d["id"]] for d in v3)
    tagged = {i for i, d in v2.items() if d.get("fewshot_passage_overlap")}
    assert tagged and not tagged & {d["id"] for d in v3}


@NEEDS
def test_task_version_2_reproduces_frozen_set(tmp_path):
    build(tmp_path, 2)
    for name, digest in V2.items():
        assert sha(tmp_path / name) == digest, name


@NEEDS
def test_task_version_3_reproduces_committed_tasks(tmp_path):
    build(tmp_path, 3)
    for name in (
        *[f"{f}.jsonl" for f in FILES],
        "locators.jsonl",
        "held_back.jsonl",
        "eval_chunk_ids.txt",
    ):
        assert sha(tmp_path / name) == sha(FROZEN / name), name
