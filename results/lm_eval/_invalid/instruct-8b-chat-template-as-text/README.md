Invalid: lm-eval 0.4.13 --apply_chat_template with tokenizer_mode=mistral renders the template to a
string and re-encodes it, so "<s>[INST]...[/INST]" reached the model as ordinary text tokens (no BOS,
no control tokens). Kept for the record; excluded from results/table.md. See notes/decisions.md.
