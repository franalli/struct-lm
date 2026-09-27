"""Group Relative Policy Optimization with TRL GRPOTrainer + verifiable rewards.

Data: JSONL of {"prompt": [...messages], "answer": "<reference>"}. Extra columns such as
`answer` are passed to every reward function as kwargs.

Reward functions return one float per completion. Keep them cheap and deterministic;
log each one separately (TRL does this per function) to catch reward hacking early,
e.g. format reward saturating while correctness stays flat.
"""

import json
import re

from common import load_model_and_tokenizer, lora_config, parse_config
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _text(completion) -> str:
    # conversational datasets yield [{"role": "assistant", "content": ...}]
    return completion[0]["content"] if isinstance(completion, list) else completion


def _parse_json(text: str):
    m = FENCE.search(text)
    try:
        return json.loads(m.group(1) if m else text)
    except (json.JSONDecodeError, TypeError):
        return None


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion parses as JSON, else 0.0."""
    return [1.0 if _parse_json(_text(c)) is not None else 0.0 for c in completions]


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """Field-level F1 between predicted and reference JSON objects; exact match otherwise."""
    rewards = []
    for c, ref in zip(completions, answer):
        pred, gold = _parse_json(_text(c)), _parse_json(ref)
        if isinstance(pred, dict) and isinstance(gold, dict) and gold:
            hits = sum(pred.get(k) == v for k, v in gold.items())
            p, r = hits / max(1, len(pred)), hits / len(gold)
            rewards.append(0.0 if hits == 0 else 2 * p * r / (p + r))
        else:
            rewards.append(1.0 if _text(c).strip() == str(ref).strip() else 0.0)
    return rewards


def length_penalty(completions, **kwargs) -> list[float]:
    """Mild penalty past 512 chars, to discourage padding the answer."""
    return [-min(1.0, max(0, len(_text(c)) - 512) / 2048) for c in completions]


REWARDS = {f.__name__: f for f in (format_reward, correctness_reward, length_penalty)}


def main() -> None:
    cfg = parse_config()
    model, tok = load_model_and_tokenizer(cfg["model"])
    ds = load_dataset("json", data_files={"train": cfg["data"]["train"]})["train"]

    names = cfg["rewards"]["functions"]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[REWARDS[n] for n in names],
        args=GRPOConfig(**cfg["training"], reward_weights=cfg["rewards"].get("weights")),
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
