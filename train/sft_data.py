"""Stage 3 SFT records as the token ids the trainer sees: pre-tokenised in Mistral's chat format.

A record {"prompt": [user], "completion": [assistant]} renders through mistral-common exactly as the
KPI eval's --chat path does: vLLM's llm.chat with tokenizer_mode="mistral" encodes the prompt with
encode_chat_completion, and eval/gold_lp.py (whose token_ids() this calls) the prompt plus answer
with encode_instruct. So the training sequence is
  input_ids        <s> [INST] prompt [/INST] completion </s>   = [1, 3, ..., 4, ..., 2]
  completion_mask  0 through [/INST], 1 from the first answer token through </s>
with no system prompt (the base tokenizer has none, and none is passed) and no image tokens.
The trainer (train/sft.py) gets these columns as they are: TRL's dataset preparation is off, and its
collator turns completion_mask into labels (-100 on the prompt and the padding). Why not TRL's
own chat path: TRL 0.29 has no Mistral backend, and transformers' MistralCommonBackend refuses a
prompt + completion conversation in its default mode and silently drops the assistant mask
(notes/decisions.md, Stage 3b pre-registration). tests/test_template.py checks these tensors.

  .venv/bin/python train/sft_data.py template-prompts   # -> eval/sft_template_prompts.jsonl

The template prompts (one record per format) go to vLLM in the smoke run, whose prompt_token_ids
the test compares with encode()'s.
"""

import hashlib
import json
import sys
from pathlib import Path

# eval/gold_lp.py: locally next to train/, in the Modal training image under /root/eval
for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "gold_lp.py").exists():
        sys.path.insert(0, str(_d))
        break
from gold_lp import token_ids, tokenizer

MAX_TOKENS = 4096
FORMATS = ("closed_book", "grounded", "abstain", "definition", "replay")
REPO = Path(__file__).resolve().parents[1]
TEMPLATE_PROMPTS = REPO / "eval/sft_template_prompts.jsonl"


def texts(rec: dict) -> tuple[str, str]:
    """The record's user prompt and assistant completion (single-turn by construction)."""
    (p,), (c,) = rec["prompt"], rec["completion"]
    if p["role"] != "user" or c["role"] != "assistant":
        raise ValueError(f"{rec.get('eid')}: expected one user and one assistant message")
    return p["content"], c["content"]


def encode(rec: dict, model: str) -> dict:
    """{input_ids, completion_mask} for one record, through mistral-common. `model` names the
    tokenizer (a checkpoint dir with tekken.json, or the hub id); every checkpoint here carries the
    base's byte-identical tekken.json (merge.py checks it)."""
    prompt, answer = texts(rec)
    e = token_ids(model, True, prompt, answer)
    if e is None:
        raise ValueError(f"{rec.get('eid')}: prompt ids are not a prefix of prompt + completion")
    ids, n_prompt = e
    if len(ids) > MAX_TOKENS:  # dropped at assembly, never truncated: this would be a new record
        raise ValueError(f"{rec.get('eid')}: {len(ids)} tokens > {MAX_TOKENS}")
    return {"input_ids": ids, "completion_mask": [0] * n_prompt + [1] * (len(ids) - n_prompt)}


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_hash(sft_dir: str | Path) -> str:
    """Check every file listed in <sft_dir>/SHA256SUMS against its digest and return the dataset
    hash: the sha256 of SHA256SUMS itself (it covers train.jsonl and sft_val.jsonl). Training runs
    cite it; a set that doesn't match its frozen digests is refused."""
    sft_dir = Path(sft_dir)
    sums = sft_dir / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, name = line.split()
        if sha256(sft_dir / name) != digest:
            raise SystemExit(f"{sft_dir / name} doesn't match SHA256SUMS: not the frozen SFT set")
    return sha256(sums)


def special_ids(model: str) -> dict[str, int]:
    """Control-token ids by name ("[INST]", "[IMG]", ...), from the tokenizer, not hard-coded."""
    tek = tokenizer(model).instruct_tokenizer.tokenizer
    return {t["token_str"]: t["rank"] for t in tek._all_special_tokens}


def template_records(train: list[dict]) -> list[dict]:
    """The first train record of each format (file order is a fixed hash order): the five records
    B1 renders through both paths."""
    first = {}
    for r in train:
        first.setdefault(r["format"], r)
    return [first[f] for f in FORMATS]


def main() -> None:
    if sys.argv[1:] != ["template-prompts"]:
        raise SystemExit(__doc__)
    recs = template_records(load(REPO / "data/sft/train.jsonl"))
    with TEMPLATE_PROMPTS.open("w") as f:
        for r in recs:
            row = {"id": r["eid"], "format": r["format"], "prompt": texts(r)[0]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(recs)} prompts -> {TEMPLATE_PROMPTS}")


if __name__ == "__main__":
    main()
