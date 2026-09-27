"""Direct Preference Optimization with TRL DPOTrainer.

Data: JSONL of {"prompt": [...messages], "chosen": [...messages], "rejected": [...messages]}.
Pairs can come from judge.py scoring N samples of the SFT model (best vs worst), which
keeps preferences on-policy.

With LoRA, ref_model=None makes TRL use the adapter-disabled base as the frozen
reference, so no second model copy sits in memory.
"""

from common import load_model_and_tokenizer, lora_config, parse_config
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer


def main() -> None:
    cfg = parse_config()
    model, tok = load_model_and_tokenizer(cfg["model"])
    ds = load_dataset(
        "json", data_files={"train": cfg["data"]["train"], "validation": cfg["data"]["validation"]}
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**cfg["training"]),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
