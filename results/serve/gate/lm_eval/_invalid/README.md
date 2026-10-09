# Excluded gate runs

`dpo-strict-fp8kv-without-fp8-kv/`: the first GSM8K run of the FP8-KV variant (2026-10-09). The
wrapper (`eval/gsm8k_gate.py`) didn't take `--kv-cache-dtype` yet, so it ran the FP8 checkpoint with
a bf16 KV cache: it is FP8's line, not FP8-KV's (the same 0.8074). The rerun with the flag was
refused by the Modal spend limit, so FP8-KV's GSM8K cell is blank (`notes/decisions.md`, Stage 6
outcome). Kept out of `gate/lm_eval/dpo-strict-fp8kv/` so `train/report.py` never reads it.
