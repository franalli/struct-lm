"""Continued pre-training (LoRA) on the packed domain corpus.

Plain next-token loss on every token. Watch two numbers: domain val loss (should fall)
and eval/run_lm_eval.sh general benchmarks (should not fall much). That trade-off is the
headline of this stage.
"""

from common import load_model_and_tokenizer, lora_config, parse_config
from datasets import load_from_disk
from peft import get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def main() -> None:
    cfg = parse_config()
    model, tok = load_model_and_tokenizer(cfg["model"])
    if (peft_cfg := lora_config(cfg)) is not None:
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    ds = load_from_disk(cfg["data"]["packed"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**cfg["training"]),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    trainer.save_model()
    tok.save_pretrained(cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
