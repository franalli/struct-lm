"""Stage 5 data: the candidate tasks (data/grpo/tasks.jsonl) and the probe's window
(data/grpo/{train,val}.jsonl). A failure blocks every GRPO launch.

  - every task is a Stage 4 pool prompt outside the judge split, one user message, with a verifier
    the reward knows, and fits the rollout engine (prompt + 256 <= vllm_max_model_length);
  - the window: every train / val task is a candidate task, train and val share no fact (every
    paraphrase of a held-out fact is held out with it), and the files match SHA256SUMS;
  - each task's verifier is rebuilt from its SFT builder record exactly (no hand edits).
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
GRPO = REPO / "data/grpo"
BASE = "mistralai/Ministral-3-8B-Base-2512"

TASKS = pytest.mark.skipif(not (GRPO / "tasks.jsonl").exists(), reason="no data/grpo/tasks.jsonl")
WINDOW = pytest.mark.skipif(not (GRPO / "train.jsonl").exists(), reason="no data/grpo/train.jsonl")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def tasks():
    return jsonl(GRPO / "tasks.jsonl")


@TASKS
def test_tasks_are_pool_prompts(tasks):
    pool = {r["id"]: r for r in jsonl(REPO / "data/dpo/prompts.jsonl")}
    assert Counter(t["format"] for t in tasks) == {
        "closed_book": 1084,
        "grounded": 359,
        "abstain": 176,
    }
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for t in tasks:
        p = pool[t["id"]]
        assert p["dpo_split"] != "judge", t["id"]  # the win-rate prompts stay unseen
        assert t["prompt"] == p["prompt"] and len(t["prompt"]) == 1, t["id"]
        assert t["prompt"][0]["role"] == "user"
        assert t["verifier"]["type"] == {"closed_book": "closed_book", "grounded": "cite",
                                         "abstain": "abstain"}[t["format"]]  # fmt: skip


@TASKS
def test_verifiers_rebuild_from_the_records(tasks):
    sys.path.append(str(REPO / "train"))
    from grpo_rewards import verifier_for

    recs = {}
    for line in (REPO / "data/sft/work/judged.jsonl").open():
        r = json.loads(line)
        recs[r["eid"]] = r
    for t in tasks:
        assert t["verifier"] == verifier_for(t["format"], recs[t["id"]]), t["id"]
        v = t["verifier"]
        if v["type"] in ("cite", "abstain"):  # the prompt shows every passage the rule accepts
            assert all(f"[{c}]" in t["prompt"][0]["content"] for c in v["chunk_ids"]), t["id"]


@TASKS
def test_prompts_fit_the_rollout_engine(tasks):
    from huggingface_hub import try_to_load_from_cache
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    cached = try_to_load_from_cache(BASE, "tekken.json")
    if not isinstance(cached, str):
        pytest.skip(f"no cached {BASE} tokenizer")
    tok = MistralTokenizer.from_file(cached)
    t = yaml.safe_load((REPO / "train/configs/grpo.yaml").read_text())["training"]
    room = t["vllm_max_model_length"] - t["max_completion_length"]
    longest = max(
        len(tok.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=x["prompt"][0]["content"])])
        ).tokens)
        for x in tasks
    )  # fmt: skip
    assert longest <= room, (longest, room)


@WINDOW
def test_window_split_and_hash(tasks):
    sys.path.insert(0, str(REPO / "train"))
    import sft_data

    sft_data.dataset_hash(GRPO)  # SystemExit on any mismatch
    by_id = {t["id"]: t for t in tasks}
    train, val = jsonl(GRPO / "train.jsonl"), jsonl(GRPO / "val.jsonl")
    assert len(val) == 50
    for r in train + val:
        assert {k: r[k] for k in by_id[r["id"]]} == by_id[r["id"]], r["id"]  # a candidate, unedited
        assert 0 < r["probe_correct"] < r["probe_n"], r["id"]  # in the window
    assert not {r["fact_id"] for r in train} & {r["fact_id"] for r in val}
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert {r["format"] for r in val} == {"closed_book", "grounded", "abstain"}
    meta = json.loads((GRPO / "tasks_meta.json").read_text())["probe"]
    assert meta["train_total"] == len(train) and meta["val_total"] == len(val)
    assert meta["dataset_hash"] == hashlib.sha256((GRPO / "SHA256SUMS").read_bytes()).hexdigest()
