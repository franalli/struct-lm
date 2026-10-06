"""B1 (Stage 3b): the SFT trainer's input is the eval's chat format, and the loss lands on the answer.

Checked on the exact tensors SFTTrainer receives: train/sft_data.encode() output (the pre-tokenised
dataset) through TRL 0.29's DataCollatorForLanguageModeling with completion_only_loss, the collator
SFTTrainer builds for it. Five records, one per format, in full detail; the shape and mask
invariants on all 2,516. The vLLM half (the eval's llm.chat renders the same prompt ids) reads the
smoke run's template samples when they have been pulled (results/runs/smoke-sft/samples/).

A failure here blocks every Stage 3 launch (notes/decisions.md, Stage 3b pre-registration)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
SFT = REPO / "data/sft"
BASE = "mistralai/Ministral-3-8B-Base-2512"
SMOKE_SAMPLES = REPO / "results/runs/smoke-sft/samples/template.jsonl"

HAVE = pytest.mark.skipif(not (SFT / "train.jsonl").exists(), reason="no data/sft/train.jsonl")


@pytest.fixture(scope="module")
def sd():
    import sft_data

    return sft_data


@pytest.fixture(scope="module")
def train(sd):
    return sd.load(SFT / "train.jsonl")


@pytest.fixture(scope="module")
def every(sd, train):
    return train + sd.load(SFT / "sft_val.jsonl")


@pytest.fixture(scope="module")
def special(sd):
    return sd.special_ids(BASE)


@pytest.fixture(scope="module")
def tek(sd):
    return sd.tokenizer(BASE).instruct_tokenizer.tokenizer


def eval_prompt_ids(prompt: str) -> list[int]:
    """The KPI eval's --chat rendering: vLLM's llm.chat under tokenizer_mode="mistral" is
    mistral-common's encode_chat_completion of the user turn (run_eval.generate)."""
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_hf_hub(BASE)
    req = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    return tok.encode_chat_completion(req).tokens


def collate(encoded: list[dict]):
    """SFTTrainer's collator for this dataset: pad id from the tokenizer sft.py passes."""
    pytest.importorskip("trl")
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    pad = AutoTokenizer.from_pretrained(BASE).pad_token_id
    return DataCollatorForLanguageModeling(pad_token_id=pad, completion_only_loss=True)(encoded)


def check_shape(enc: dict, special: dict, num_special: int) -> int:
    """[1, 3, ..., 4, ..., 2], mask 0 through [/INST] and 1 after; returns the [/INST] index."""
    ids, mask = enc["input_ids"], enc["completion_mask"]
    end_inst = mask.index(1) - 1
    assert ids[:2] == [special["<s>"], special["[INST]"]]
    assert ids[end_inst] == special["[/INST]"]
    assert ids[-1] == special["</s>"]
    assert mask == [0] * (end_inst + 1) + [1] * (len(ids) - end_inst - 1)
    assert len(ids) - end_inst - 1 >= 2  # at least one answer token, then </s>
    # every other id is text: Tekken puts all control tokens (system prompt, images, tools, a
    # second [INST]) below num_special, and text never encodes to them
    body = ids[2:end_inst] + ids[end_inst + 1 : -1]
    assert min(body) >= num_special
    return end_inst


@HAVE
def test_five_formats_in_detail(sd, train, special, tek):
    recs = sd.template_records(train)
    assert [r["format"] for r in recs] == list(sd.FORMATS)
    encoded = [sd.encode(r, BASE) for r in recs]
    batch = collate(encoded)
    # text path only: no pixel_values or image sizes reach the model
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    for i, (r, enc) in enumerate(zip(recs, encoded)):
        prompt, completion = sd.texts(r)
        ids = enc["input_ids"]
        end_inst = check_shape(enc, special, tek.num_special_tokens)
        # prompt ids: the eval's chat path, and transformers' backend as a third witness
        assert ids[: end_inst + 1] == eval_prompt_ids(prompt), r["format"]
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(BASE, mode="agnostic")
        witness = hf.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True)
        assert ids[: end_inst + 1] == witness["input_ids"], r["format"]
        # no system prompt, no image tokens anywhere
        for name in ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"):
            assert special[name] not in ids, name
        # labels: -100 on the prompt and the padding, the ids on the answer and </s>
        n = len(ids)
        labels = batch["labels"][i].tolist()
        assert labels[: end_inst + 1] == [-100] * (end_inst + 1)
        assert labels[end_inst + 1 : n] == ids[end_inst + 1 :]
        assert set(labels[n:]) <= {-100}
        assert batch["attention_mask"][i].tolist() == [1] * n + [0] * (len(labels) - n)
        # the masked span is the completion, exactly, then </s>
        span = [t for t, m in zip(ids, enc["completion_mask"]) if m]
        assert span[-1] == special["</s>"]
        assert tek.decode(span[:-1]) == completion, r["format"]


@HAVE
def test_all_records(sd, every, special, tek):
    for r in every:
        enc = sd.encode(r, BASE)
        check_shape(enc, special, tek.num_special_tokens)
        assert len(enc["input_ids"]) == r["n_tokens"] <= sd.MAX_TOKENS, r["eid"]


def test_trainer_gets_a_tokenizer_not_a_processor():
    """sft.py passes common.load_model_and_tokenizer's AutoTokenizer as processing_class: a
    processor would make TRL treat the model as a VLM and expect images."""
    from transformers import AutoTokenizer, ProcessorMixin

    tok = AutoTokenizer.from_pretrained(BASE)
    assert not isinstance(tok, ProcessorMixin)
    assert tok.pad_token_id == 11 and tok.eos_token_id == 2


@HAVE
@pytest.mark.skipif(not SMOKE_SAMPLES.exists(), reason="smoke-sft template samples not pulled")
def test_vllm_renders_the_training_prompt_ids(sd, train):
    """The smoke run sent the five template prompts through run_eval.make_llm + llm.chat (the
    KPI eval's --chat path, on the merged smoke checkpoint): vLLM's prompt ids are the trainer's."""
    by_id = {r["eid"]: r for r in sd.template_records(train)}
    rows = [json.loads(line) for line in SMOKE_SAMPLES.read_text().splitlines()]
    assert {row["id"] for row in rows} == set(by_id)
    for row in rows:
        enc = sd.encode(by_id[row["id"]], BASE)
        n_prompt = enc["completion_mask"].index(1)
        assert row["prompt_token_ids"] == enc["input_ids"][:n_prompt], row["id"]
