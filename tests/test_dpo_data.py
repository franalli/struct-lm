"""Stage 4 data: the prompt pool (data/dpo/prompts.jsonl) and the preference pairs
(data/dpo/{train,val}.jsonl). The pair tests are step 7 of the Stage 4 plan: the trainer gets
exactly the tokens the eval renders, and a failure blocks every DPO launch (notes/decisions.md,
2026-10-08 pre-registration).

Pairs are pre-tokenised: `prompt_ids` and the completions' ids are vLLM's own (the exact on-policy
tokens); train/dpo.py passes them through TRL's DataCollatorForPreference untouched."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
DPO = REPO / "data/dpo"
SFT = REPO / "data/sft"
BASE = "mistralai/Ministral-3-8B-Base-2512"
FORMATS = ("closed_book", "definition", "grounded", "abstain")

POOL = pytest.mark.skipif(not (DPO / "prompts.jsonl").exists(), reason="no data/dpo/prompts.jsonl")
PAIRS = pytest.mark.skipif(not (DPO / "train.jsonl").exists(), reason="no data/dpo/train.jsonl")
# the as-run set (dpo, dpo-seed1, dpo-2ep) and the strict rebuild (dpo-strict, 2026-10-09)
PAIR_DIRS = [d for d in (DPO, DPO / "strict") if (d / "train.jsonl").exists()]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def pool():
    return jsonl(DPO / "prompts.jsonl")


@pytest.fixture(scope="module", params=PAIR_DIRS, ids=lambda d: d.name)
def pair_dir(request):
    return request.param


@pytest.fixture(scope="module")
def pairs(pair_dir):
    return {s: jsonl(pair_dir / f"{s}.jsonl") for s in ("train", "val")}


@pytest.fixture(scope="module")
def special():
    import sft_data

    return sft_data.special_ids(BASE)


# ---- the pool ----


@POOL
def test_pool_schema_and_splits(pool):
    assert {r["format"] for r in pool} == set(FORMATS)  # no replay (2026-10-08)
    ids = [r["id"] for r in pool]
    assert len(ids) == len(set(ids))
    texts = [r["prompt"][0]["content"] for r in pool]
    assert len(texts) == len(set(texts))
    assert all(len(r["prompt"]) == 1 and r["prompt"][0]["role"] == "user" for r in pool)
    split = Counter(r["dpo_split"] for r in pool)
    assert split["judge"] == 100 and set(split) == {"train", "val", "judge"}
    for f in FORMATS:  # every split holds every format (stratified)
        assert {r["dpo_split"] for r in pool if r["format"] == f} == {"train", "val", "judge"}


@POOL
def test_pool_head_is_the_probe_set(pool):
    """The pre-registered probe (2026-10-06) is the first 20 prompts of the pool as it was then:
    the first 20 distinct domain prompts of the SFT train set, in its order."""
    head, seen = [], set()
    for r in jsonl(SFT / "train.jsonl"):
        if r["format"] == "replay" or r["prompt"][0]["content"] in seen:
            continue
        seen.add(r["prompt"][0]["content"])
        head.append(r["eid"])
        if len(head) == 20:
            break
    assert [r["id"] for r in pool[:20]] == head
    meta = json.loads((DPO / "prompts_meta.json").read_text())
    assert meta["probe_ids"] == head
    assert meta["sha256"] == hashlib.sha256((DPO / "prompts.jsonl").read_bytes()).hexdigest()


def make_guard():
    """sft_guard from data/scripts, without leaving data/scripts/common.py where train/'s
    `import common` would find it (tests/test_dpo_train.py imports train/dpo.py after this)."""
    path = str(REPO / "data/scripts")
    sys.path.insert(0, path)
    try:
        from sft_guard import Guard

        return Guard()
    finally:
        sys.path.remove(path)
        m = sys.modules.get("common")
        if m is not None and path in (getattr(m, "__file__", "") or ""):
            del sys.modules["common"]


@POOL
def test_pool_guard_clean(pool):
    guard = make_guard()
    for r in pool:
        assert all(guard.allowed(c) for c in r["source_chunks"]), r["id"]
    sft_prompts = {r["prompt"][0]["content"] for r in jsonl(SFT / "train.jsonl")}
    sft_val = {r["prompt"][0]["content"] for r in jsonl(SFT / "sft_val.jsonl")}
    for r in pool:
        text = r["prompt"][0]["content"]
        assert text not in sft_val, r["id"]
        if r["source"] == "sft_train":
            assert text in sft_prompts, r["id"]
        else:  # the definitions the Stage 3 cap cut: never trained on
            assert r["format"] == "definition" and text not in sft_prompts, r["id"]


# ---- the pairs (step 7: blocks launch) ----


def sft_prompt_ids(prompt: str) -> list[int]:
    """The SFT trainer's and the KPI eval's --chat rendering of a user turn."""
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_hf_hub(BASE)
    return tok.encode_chat_completion(
        ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    ).tokens


def five_per_format(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if len(by[r["format"]]) < 5:
            by[r["format"]].append(r)
    return [r for f in FORMATS for r in by[f]]


@PAIRS
def test_dataset_hash_and_splits(pairs, pool, pair_dir):
    import sft_data

    sft_data.dataset_hash(pair_dir)  # SystemExit on any mismatch
    split = {r["id"]: r["dpo_split"] for r in pool}
    for name, rows in pairs.items():
        assert rows, name
        assert {split[r["prompt_id"]] for r in rows} == {name}  # no val or judge prompt in train
    ids = [r["id"] for rows in pairs.values() for r in rows]
    assert len(ids) == len(set(ids))


@PAIRS
def test_records_in_detail(pairs, special):
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tek = MistralTokenizer.from_hf_hub(BASE).instruct_tokenizer.tokenizer
    n_special = tek.num_special_tokens
    bos, eos = special["<s>"], special["</s>"]
    checked = five_per_format(pairs["train"]) + five_per_format(pairs["val"])
    assert {r["format"] for r in checked} == {r["format"] for r in pairs["train"]}
    for r in checked:
        (p,), (c,), (j,) = r["prompt"], r["chosen"], r["rejected"]
        assert (p["role"], c["role"], j["role"]) == ("user", "assistant", "assistant")
        assert r["system_variant"] is None
        ids = r["prompt_ids"]
        assert ids == sft_prompt_ids(p["content"]), r["id"]  # the SFT / eval render
        assert ids[:2] == [bos, special["[INST]"]] and ids[-1] == special["[/INST]"]
        assert ids.count(bos) == 1
        assert special["[SYSTEM_PROMPT]"] not in ids and special["[/SYSTEM_PROMPT]"] not in ids
        for key, msg in (("chosen_ids", c), ("rejected_ids", j)):
            comp = r[key]
            assert comp[-1] == eos and comp.count(eos) == 1, (r["id"], key)
            assert min(comp[:-1]) >= n_special, (r["id"], key)  # no control token inside
            assert tek.decode(comp[:-1]).strip() == msg["content"].strip(), (r["id"], key)
        assert r["chosen_ids"] != r["rejected_ids"]
        assert r["score_chosen"] > r["score_rejected"]


@PAIRS
def test_lengths_and_scores(pairs):
    for rows in pairs.values():
        for r in rows:
            longer = max(len(r["chosen_ids"]), len(r["rejected_ids"]))
            assert len(r["prompt_ids"]) + longer <= 4096, r["id"]
            assert r["label_source"] in ("verifier", "judge", "rule")
            assert r["format"] in FORMATS


@PAIRS
def test_collator_builds_prompt_plus_completion(pairs):
    """TRL's DataCollatorForPreference, as DPOTrainer builds it for this dataset: chosen half
    first, each row prompt + completion, completion_mask exactly over the completion."""
    pytest.importorskip("trl")
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    rows = five_per_format(pairs["train"])[:4]
    feats = [{k: r[k] for k in ("prompt_ids", "chosen_ids", "rejected_ids")} for r in rows]
    batch = DataCollatorForPreference(pad_token_id=11)(feats)
    n = len(rows)
    for k, r in enumerate(rows):
        for half, key in ((0, "chosen_ids"), (1, "rejected_ids")):
            want = r["prompt_ids"] + r[key]
            got = batch["input_ids"][half * n + k][: len(want)].tolist()
            assert got == want
            mask = batch["completion_mask"][half * n + k][: len(want)].tolist()
            assert mask == [0] * len(r["prompt_ids"]) + [1] * len(r[key])


@pytest.mark.skipif(not (DPO / "strict/train.jsonl").exists(), reason="no data/dpo/strict")
def test_strict_labels():
    """dpo-strict's closed-book labels are the strict checker's: every chosen answer passes it
    (one line, scorers.qa_strict), every rejected one fails it."""
    sys.path.insert(0, str(REPO / "eval"))
    from scorers import answer_line, qa_strict

    gold = {}
    for line in (SFT / "work/judged.jsonl").open():
        r = json.loads(line)
        gold[r["eid"]] = (r["gold"], r["kind"])
    ok = lambda t, g: (
        len([x for x in t.splitlines() if x.strip()]) == 1 and qa_strict(answer_line(t), *g)
    )
    rows = [r for s in ("train", "val") for r in jsonl(DPO / f"strict/{s}.jsonl")]
    cb = [r for r in rows if r["format"] == "closed_book"]
    assert cb
    failed = set()
    for r in cb:
        g = gold[r["prompt_id"]]
        if not ok(r["chosen"][0]["content"], g):
            failed.add(r["id"])
        assert not ok(r["rejected"][0]["content"], g), r["id"]
    # the set was frozen before the 2026-10-09 comma fix, which rejects one chosen answer: a right
    # answer adding a noun to a gold that holds "or" (notes/decisions.md)
    assert failed == {"usace-em-1110-2-1611:p62:c0:f2:closed_book:1#0"}
    meta = json.loads((DPO / "strict/pairs_meta.json").read_text())
    assert meta["closed_book_verifier"] == "scorers.qa_strict"
