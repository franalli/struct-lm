"""make_tasks.py --task-version 1 still rebuilds the eval frozen on 2026-09-27 byte for byte (the
130-item domain_qa behind results/table_v1.md), and --task-version 2 (the default) rebuilds the
committed eval/tasks/. Runs from the LLM disk cache, so it makes no API calls when the cache is
complete; skipped where the cache, chunks or key aren't available."""

import hashlib
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


NEEDS = pytest.mark.skipif(
    not (REPO / "eval/.cache/llm_cache.jsonl").exists()
    or not (REPO / "data/processed/chunks.jsonl").exists()
    or not os.environ.get("MISTRAL_API_KEY"),
    reason="needs the LLM cache, data/processed/chunks.jsonl and MISTRAL_API_KEY",
)


def build(out: Path, version: int) -> None:
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


@NEEDS
def test_task_version_2_reproduces_committed_tasks(tmp_path):
    build(tmp_path, 2)
    for name in (
        *[f"{f}.jsonl" for f in FILES],
        "locators.jsonl",
        "held_back.jsonl",
        "eval_chunk_ids.txt",
    ):
        assert sha(tmp_path / name) == sha(FROZEN / name), name
