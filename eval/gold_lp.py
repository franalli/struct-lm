"""Gold-answer log-probability for domain_qa: how much probability a checkpoint gives the gold
answer, a continuous companion to qa_acc on the same items, with no new items to review.
run_eval.py computes it in the generation engine (one model load) and saves it per item in
generations.jsonl; score() turns it into the gold_lp columns.

Why: qa_acc is pass/fail, so a model that half-knows a fact (the gold became more likely, but
isn't yet its top answer) scores 0. The gold's log-probability sees that, item by item, with no
parser and no judge, and far less variance than accuracy.

The prompt is exactly the one generation uses (prompts.qa_prompt), and the scored continuation is
the gold answer as the model would write it:
  base   " " + answer + "\\n"   (the few-shot format "A: 0.95 d_b\\n"; the newline ends the answer,
                                so a gold that is a prefix of a longer string isn't free)
  chat   the assistant message `answer`, then EOS (mistral-common's encode_instruct); scored in
         the chat format, so an Instruct row is not comparable to base-format rows
The answer's tokens are the full sequence minus the prompt's, which needs the prompt's tokens to be
a prefix of prompt + answer: Tekken could merge across the boundary (the space after "A:").
gold_token_ids() checks every item and returns the failures instead of scoring wrong tokens.
"""

from pathlib import Path

_TOKENIZERS: dict = {}


def tokenizer(model: str):
    """The checkpoint's Tekken tokenizer: its own tekken.json (merged checkpoints carry the base's
    byte-identical copy) or the hub repo's, as vLLM loads it with tokenizer_mode="mistral"."""
    if model not in _TOKENIZERS:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        local = Path(model) / "tekken.json"
        _TOKENIZERS[model] = (
            MistralTokenizer.from_file(str(local))
            if local.exists()
            else MistralTokenizer.from_hf_hub(model)
        )
    return _TOKENIZERS[model]


def token_ids(model: str, chat: bool, prompt: str, answer: str) -> tuple[list[int], int] | None:
    """(prompt + answer token ids, number of prompt tokens), or None if the prompt's tokens are
    not a prefix of the full sequence."""
    from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest

    tok = tokenizer(model)
    if chat:
        p = tok.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        ).tokens
        full = tok.instruct_tokenizer.encode_instruct(
            InstructRequest(
                messages=[UserMessage(content=prompt), AssistantMessage(content=answer)]
            )
        ).tokens
    else:
        tek = tok.instruct_tokenizer.tokenizer
        p = [tek.bos_id, *tek.encode(prompt, bos=False, eos=False)]
        full = [tek.bos_id, *tek.encode(f"{prompt} {answer}\n", bos=False, eos=False)]
    return (full, len(p)) if full[: len(p)] == p and len(full) > len(p) else None


def gold_token_ids(model: str, chat: bool, items: list[dict]) -> tuple[dict, list[str]]:
    """{item id: (ids, n_prompt)} for every domain_qa work item (run_eval's {id, prompt, ref}),
    and the ids whose boundary check failed (left out, never scored on the wrong tokens)."""
    encoded, failed = {}, []
    for it in items:
        e = token_ids(model, chat, it["prompt"], it["ref"]["answer"])
        if e is None:
            failed.append(it["id"])
        else:
            encoded[it["id"]] = e
    return encoded, failed
