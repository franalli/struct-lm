# Contributions

Rough edges hit while building this pipeline: library behaviour that silently does the wrong
thing for this workload, and what the repo does about it. Each entry is a candidate upstream issue
or PR. Newest at the bottom. Versions are the ones pinned in `uv.lock` and the Modal images.

Template:

```
## YYYY-MM-DD: <library>: <one-line symptom>
**Where:** package version, file:line
**What happens:** the behaviour, and why it bites here
**Repro:** the smallest thing that shows it
**Here:** the workaround in this repo
**Upstream:** the issue or PR it suggests; status
```

---

## 2026-09-27: TRL: default packing keeps only the first `max_length` tokens of each document
**Where:** trl 0.29.1, `SFTConfig.packing_strategy` defaults to `"bfd"` (`trl/trainer/sft_config.py:185`);
`trl/data_utils.py:680-693` keeps one fragment per row unless `requeue_truncated_sequences`.
**What happens:** best-fit-decreasing packing truncates any sequence longer than `max_length`. The
help string says so ("truncates overflow"), but nothing warns at run time. For continued
pre-training on whole documents it discards almost everything: the 234 train documents average
~83k tokens, so `packing=True, max_length=4096` would train on ~1M of the corpus's 19.4M tokens,
and the loss curve would look normal.
**Repro:** `SFTTrainer(args=SFTConfig(packing=True, max_length=4096), train_dataset=<one 20k-token text>)`
yields one 4,096-token row.
**Here:** `train/packing.py` builds the windows itself (BOS + document + EOS, concatenated, cut at
4,096), and `cpt.py` fails if the window count differs from what the files' token counts imply.
**Upstream:** warn when packing truncates more than a small fraction of tokens, pointing at
`"wrapped"` / `"bfd-requeue"` for pre-training data. Not filed.

## 2026-09-27: TRL + Mistral Tekken: the EOS appended to raw text is tokenised as text
**Where:** trl 0.29.1 `trl/trainer/sft_trainer.py:1026-1028` (`add_eos` appends the `eos_token`
string to `text`); transformers 5.16.1 `MistralCommonBackend` over mistral-common 1.12.0 Tekken,
whose text encoder has no special tokens.
**What happens:** for Ministral 3, `tokenizer("Load factor." + tokenizer.eos_token)` gives
`[1, 9373, 7342, 15342, 1115, 1062]`, decoded `<s> Load factor .</ s >`: no id 2. Every document
boundary becomes the literal characters `</s>`, so the model learns to write them and never sees
a real EOS.
**Repro:** the call above (checked 2026-09-27 on the Mac).
**Here:** documents are tokenised by `train/packing.py` with BOS/EOS added as ids, and SFTTrainer
gets `input_ids` with `dataset_kwargs={"skip_prepare_dataset": True}`.
**Upstream:** append `eos_token_id` after tokenisation instead of the string before it (or check
that the string round-trips to one id). Not filed.

## 2026-09-27: TRL: `SFTConfig.max_length` defaults to 1024, also for pre-tokenised data
**Where:** trl 0.29.1 `trl/trainer/sft_config.py:166-167`.
**What happens:** a dataset of pre-tokenised 4,096-token windows passed with `packing=False` and no
`max_length` is truncated to 1,024 tokens per window without a warning: three quarters of the data
gone.
**Here:** `cpt.py` sets `max_length=4096` explicitly and skips TRL's dataset preparation.
**Upstream:** warn when truncation removes tokens from an already-tokenised dataset. Not filed.

## 2026-09-27: transformers + accelerate: FSDP2 loading can tie the 8B's `lm_head` to its embeddings
**Where:** transformers 5.16.1 `Mistral3Config.tie_word_embeddings = True` by default
(`models/mistral3/configuration_mistral3.py:63`); the 8B's `config.json` doesn't set it (its
text config doesn't either) and ships a separate `lm_head`. accelerate 1.15.0 calls
`model.tie_weights()` when FSDP2 `cpu_ram_efficient_loading` is on (`utils/fsdp_utils.py:846-848, 915-916`).
**What happens:** `from_pretrained` sees two different tensors and leaves them untied, but a later
`tie_weights()` recomputes the tie from the config and replaces the trained `lm_head` with the
input embeddings. Found by reading the source; not reproduced (it would need the 2-GPU run).
**Here:** `train/common.keep_untied()` sets `tie_word_embeddings=False` when the two tensors don't
share storage, and `train/configs/fsdp2.yaml` turns `cpu_ram_efficient_loading` off.
**Upstream:** the Ministral 3 8B hub config could state `"tie_word_embeddings": false`; transformers
could derive the default from the checkpoint. Not filed.

## 2026-09-27: repo: `--override training.eval_strategy=no` became `False` (YAML 1.1 booleans)
**Where:** `train/common.parse_config`, which parsed each override value with `yaml.safe_load`.
**What happens:** PyYAML follows YAML 1.1, where `no` / `yes` / `on` / `off` are booleans, so the
2-GPU smoke run died in `SFTConfig` with "False is not a valid IntervalStrategy" (after the model
had loaded on both ranks: minutes of 2 x H100 for nothing).
**Here:** override values spelled yes/no/on/off (any case) stay strings; only true/false are
booleans. The YAML config files never hit it (`eval_strategy: steps`, `report_to: none`).
**Upstream:** none (documented PyYAML behaviour); a reminder to parse CLI overrides narrowly.

## 2026-09-27: repo: stale claims fixed while wiring Stage 2
- `train/merge.py` said `save_pretrained` writes `tokenizer.json` only; with transformers 5,
  `MistralCommonBackend.save_pretrained` also copies `tekken.json`. The docstring now describes
  what the copy step is for (`processor_config.json`, never `params.json`).
- `eval/run_lm_eval.sh` doesn't pass `limit_mm_per_prompt={"image": 0}` although CLAUDE.md rule 3
  says it's set everywhere. With text-only prompts it should change only vLLM's memory profiling,
  not scores (not verified); left unchanged so every lm-eval row runs the same script.

## 2026-09-27: transformers: `Ministral3ForCausalLM` on a Ministral 3 checkpoint loads random weights
**Where:** transformers 5.16.1; the hub checkpoints (`Ministral-3-8B-Base-2512`, `-3B-`) store the
text model under `language_model.model.*` / `language_model.lm_head.*` (the llava layout).
**What happens:** the text-only class looks for `model.*` and `lm_head.*`, finds none, and
initialises every weight randomly, with only a load-report warning: on a tiny random model with the
real config, 21 of 21 weights were "MISSING" and all 36 checkpoint tensors "UNEXPECTED". Training
through it would train a random model (loss at ln(131072) = 11.8, not ~1.9).
**Repro:** `Ministral3ForCausalLM.from_pretrained(<Ministral 3 checkpoint>, output_loading_info=True)`.
**Here:** every stage loads `Mistral3ForConditionalGeneration` (`train/common.auto_model_class`) and
trains text only; LoRA's regex is anchored on `language_model`, full fine-tuning freezes the vision
tower. This also keeps merged checkpoints in the base's layout, which vLLM loads unchanged.
**Upstream:** a key mapping from the llava layout for `Ministral3ForCausalLM` (the conversion table
already reverses it for `Mistral3ForConditionalGeneration`), or an error rather than a warning when no
checkpoint weight matches. Not filed.

## 2026-09-27: Liger-Kernel: no `ministral3` / `mistral3` patch, so no fused linear cross-entropy
**Where:** liger-kernel 0.8.3 (`liger_kernel/transformers/monkey_patch.py`): `apply_liger_kernel_to_mistral`
exists, but `MODEL_TYPE_TO_APPLY_LIGER_FN` has no entry for `ministral3` (the text model type) or
`mistral3` (the multimodal wrapper), so `use_liger_kernel=True` has nothing to attach to.
**What happens:** the 131k-vocab logits are materialised: ~26 GB at micro-batch 4 x 4,096 counting
the fp32 copies (loss upcast, accelerate's output conversion, TRL's token-accuracy copy), ~13 GB at
micro-batch 2. The LoRA runs fit anyway (51.4 GB peak measured); the 8B full-parameter run carries
it in its per-GPU estimate.
**Here:** nothing; Apple's model-agnostic cut-cross-entropy is the fallback if ablation B doesn't fit.
**Upstream:** `apply_liger_kernel_to_ministral3` (the decoder is Mistral-family: RMSNorm, RoPE,
SwiGLU, same `lm_head` + loss path), plus the `mistral3` wrapper routing to it. A plausible first
pull request. Not filed.

## 2026-09-27: transformers: `model.to(dtype)` leaves the sub-configs' dtype behind
**Where:** transformers 5.16.1 `PreTrainedModel.save_pretrained` on `Mistral3ForConditionalGeneration`.
**What happens:** a model loaded in fp32 and cast with `.to(torch.bfloat16)` saves bf16 weights with
`"dtype": "bfloat16"` at the top of `config.json` but `"float32"` in `text_config` and
`vision_config`. vLLM with `dtype="auto"` may read the text config and serve the model in fp32 (twice
the memory, different numerics from the bf16 base).
**Here:** `train/merge.py` sets `dtype` on every sub-config before saving (found while switching the
merge to fp32; the eval harness passes `dtype=bfloat16` explicitly, `serve_vllm.sh` doesn't).
**Upstream:** keep sub-config dtypes in step with the saved weights. Not filed.

## 2026-09-27: transformers: a re-saved Ministral 3 tokenizer loses its config; vLLM's processor then fails
**Where:** transformers 5.16.1 (`AutoTokenizer.from_pretrained(<adapter dir>).save_pretrained(...)`);
vLLM 0.29 `transformers_utils/processor.get_processor`.
**What happens:** the adapter directory holds only `tekken.json` from training, so the tokenizer
re-loaded from it and saved writes a stub `tokenizer_config.json` (`"tokenizer_class":
"TokenizersBackend"`, no `bos`/`eos`/`pad`/`unk`, no `added_tokens_decoder`, no `processor_class`)
and a different `tokenizer.json`, where the base ships `LlamaTokenizerFast` with the full special-token
set. When vLLM builds Ministral 3's multimodal processor (it does unless `limit_mm_per_prompt` sets
images to 0, which `run_lm_eval.sh` doesn't), `PixtralProcessor` loads that tokenizer and dies in
`add_tokens`: `TypeError: Input must be a List[Union[str, AddedToken]]`. The KPI eval and
`vllm serve` never build the processor, so only lm-eval broke; the first `cpt-8b` eval run lost ~2
minutes of 4 GPUs, and the pipeline's `gather` cancelled perplexity, KPI and latency with it.
**Here:** `train/merge.py` writes no tokenizer; it copies the base's tokenizer files and checks they are
byte-identical to the base's. The pipeline now waits for every parallel call and reports all
failures at the end.
**Upstream:** vLLM could fall back to mistral-common's tokenizer for the processor when
`tokenizer_mode="mistral"`; transformers could keep the source config when re-saving. Not filed.

## 2026-09-27: vLLM: text-only lm-eval of a merged Ministral 3 dies in the dummy-image profiling pass
**Where:** vLLM 0.29 through lm-eval 0.4.13 (`--model vllm`), on a merged `Mistral3ForConditionalGeneration`
checkpoint (HF format, no `params.json`), without `limit_mm_per_prompt`.
**What happens:** with the base's tokenizer files restored, vLLM builds the HF `PixtralProcessor` and
profiles it with a dummy 1540x1540 image; with `tokenizer_mode=mistral` the `[IMG]` placeholder text
comes back with 0 image ids against 3,025 expected: `ValueError: Mismatch in image token count
between text and input_ids`. The base checkpoint from the hub ran the same command without error in
Stage 0.
**Here:** `eval/run_lm_eval.sh` passes `limit_mm_per_prompt={"image": 0}` (CLAUDE.md rule 3 already
said so; this script was the one place it was missing), which skips the image processor entirely.
lm-eval's `key=value` model_args can't carry a nested dict (values are parsed as scalars), so the
script now passes model_args as a JSON object, which lm-eval's CLI parses whole. A base-8b control
run with the new flag (`base-8b-mm0`) checks that the scores don't move.
**Upstream:** vLLM could skip multimodal profiling when a model is only ever given text (or when
`tokenizer_mode=mistral` handles images itself). Not filed.

## 2026-09-28: vLLM: the same checkpoint family resolves to two implementations depending on its files
**Where:** vLLM 0.29 with `tokenizer_mode=mistral` (and the default `config_format="auto"`).
**What happens:** `mistralai/Ministral-3-8B-Base-2512` ships HF files and Mistral's native
`params.json` + `consolidated.safetensors`; vLLM loads it through its native implementation
(`Resolved architecture: PixtralForConditionalGeneration`). A fine-tuned, merged copy saved by
transformers has only the HF files and loads through `Mistral3ForConditionalGeneration`. Nothing
warns, and eval deltas between the two (base vs fine-tuned, the usual comparison) silently include
the implementation difference: here MMLU -3.7 and GSM8K -6.7 points against a +0.4% change in
general perplexity (measured outside vLLM).
**Here:** `config_format="hf"` everywhere vLLM is called (`run_eval.py`, `run_lm_eval.sh`,
`serve_vllm.sh`). That alone fails on the hub repo: vLLM then loads every `*.safetensors` it holds,
`consolidated.safetensors` included, and dies on its native key names ("no module or parameter
named 'layers'"). So the base is re-saved through `train/merge.py` into an HF-only directory and
evaluated from there as `base-8b-hf`.
**Upstream:** log which format `auto` picked, and warn when a repo holds both; or document that
`tokenizer_mode=mistral` also switches the model loader. Not filed.

## 2026-09-28: vLLM: YaRN from an HF config ignores mscale / mscale_all_dim (Ministral 3 runs at the wrong attention temperature)
**Where:** vLLM 0.29.0 `model_executor/layers/rotary_embedding/__init__.py` (`get_rope`, the `"yarn"`
branch keeps only extrapolation_factor, attn_factor, beta_fast, beta_slow, apply_yarn_scaling,
truncate) and `yarn_scaling_rope.py` (`mscale = yarn_get_mscale(factor) * attn_factor` unless
`apply_yarn_scaling` is False).
**What happens:** Ministral 3's HF config says `rope_type: yarn, factor: 16, mscale: 1.0,
mscale_all_dim: 1.0` (an attention factor of 1 in transformers; the native params.json says
`apply_scale: false`). vLLM drops the two mscale keys and scales cos and sin by 1.277, so the model
runs at a different attention temperature. Any HF-format copy of the model is affected, which
includes every fine-tuned checkpoint saved by transformers: on the base's val slice, perplexity 7.226
against 6.894 (native path and transformers); grounded accuracy on our eval 0.657 against 0.852.
Nothing warns.
**Repro:** `eval/vllm_ppl.py --model <HF-only copy of Ministral-3-8B-Base-2512> --config-format hf`,
with and without `--no-yarn-scale`; compare with `--config-format mistral` on the hub id.
**Here:** `train/merge.py` adds `"apply_yarn_scaling": false` to merged configs.
**Upstream:** in the `"yarn"` branch, honour mscale / mscale_all_dim the way transformers does
(attention factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)), or map them
to `apply_yarn_scaling`. A strong first vLLM issue with a one-line repro. Not filed.
