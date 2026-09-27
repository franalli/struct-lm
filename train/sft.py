"""Supervised fine-tuning with TRL SFTTrainer.

Data: JSONL, one conversation per line: {"messages": [{"role": ..., "content": ...}, ...]}
Loss is on assistant turns only (assistant_only_loss), so the model learns to answer,
not to parrot prompts.
"""

from common import load_model_and_tokenizer, lora_config, parse_config
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def main() -> None:
    cfg = parse_config()
    model, tok = load_model_and_tokenizer(cfg["model"])
    ds = load_dataset(
        "json", data_files={"train": cfg["data"]["train"], "validation": cfg["data"]["validation"]}
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**cfg["training"]),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        peft_config=lora_config(cfg),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
