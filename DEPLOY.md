# Deploying struct-lm

The served model is `stage5-final` = `checkpoints/dpo-strict`: Ministral 3 8B after CPT, SFT and
DPO on verifiable preferences (README, Stages 2-5), served text-only. The quality gate, benchmark
and cost lines are Stage 6's (`notes/decisions.md`, 2026-10-09, pre-registered before any
quantized checkpoint existed). Every latency number here was measured on one H100 80GB HBM3 (SXM),
pinned with Modal's `gpu="H100!"` and checked at runtime.

## What to deploy

**Two checkpoints pass the pre-registered quality gate, bf16 (the reference) and FP8 W8A8, and
the choice between them is the load.**
- **bf16 (`checkpoints/dpo-strict`) up to the measured 16 requests/s of open arrivals.** There it
  has the faster first token (21-40 against 39-76 ms p50) and, at 16 req/s, the better tail (E2EL
  p99 1,298 against 1,399 ms) and goodput (98.8% against 95.6%).
- **FP8 (`checkpoints/dpo-strict-fp8`) when the server runs near saturation or the KV cache binds.**
  It gives 22% more requests/s at 64 concurrent requests (bf16 saturates near 44 req/s at this mix),
  40% lower inter-token latency there, and room for 57 concurrent 8k-token sequences against 49.6.
- **Unmeasured:** open-loop loads between 16 req/s and saturation, where the crossover lies.

FP8 with an FP8 KV cache failed one gate line (strict closed-book, seen half, −4.2 against a
3.5-point floor), and INT4 W4A16 failed broadly (gold-answer log-probability about twice the floor,
GSM8K −6.1). Neither is a deployment option for this model.

<!-- deploy-variants:start -->
| variant | quality gate |
|---|---|
| bf16 | reference |
| FP8 | ships |
| FP8 + FP8 KV | fails |
| INT4 W4A16 | fails |
<!-- deploy-variants:end -->

| checkpoint | weights on disk | in GPU memory (vLLM) | digest (merge_check / quantize_meta) |
|---|---|---|---|
| `checkpoints/dpo-strict` (bf16) | 17.8 GB | 15.94 GiB | `e686ca7bf11e…` |
| `checkpoints/dpo-strict-fp8` (FP8 W8A8: per-channel weights, per-token dynamic activations) | 10.42 GB | 9.01 GiB | `a241f60fe9e3…` |
| `checkpoints/dpo-strict-w4a16` (INT4 W4A16: GPTQ, group 128, symmetric; fails the gate) | 6.83 GB | 5.66 GiB | `c7a87b497b24…` |

The vision tower (0.43B parameters) is on disk in every variant and never loaded: with
`--limit-mm-per-prompt '{"image": 0}'` vLLM skips it (15.94 GiB loaded of 16.6 GiB on disk for bf16).
`lm_head`, the embeddings, the tower and the projector stay bf16 in the quantized checkpoints.

## Environment

```
vllm==0.29.0
transformers>=5.10.4,<5.17   # 5.17 renamed PixtralRotaryEmbedding; vLLM 0.29 imports the old name
mistral-common>=1.8.6        # Tekken tokenizer for --tokenizer-mode mistral
```

This is the eval image's pin set (`eval/modal_app.py`); `uv.lock`'s `serve` extra resolves vLLM 0.30,
which is not what was measured. FP8 W8A8 needs compute capability >= 8.9 (Ada, Hopper); vLLM selected
`CutlassFP8ScaledMMLinearKernel` on the H100 (DeepGEMM isn't importable without a CUDA toolkit in
the image, and isn't needed). INT4 W4A16 runs on compute capability >= 8.0.

## Serve

```bash
bash serve/serve_vllm.sh /path/to/dpo-strict        # bf16: moderate load
bash serve/serve_vllm.sh /path/to/dpo-strict-fp8    # FP8: near saturation, or memory-bound
```

which runs (FP8 shown; bf16 is the same with its own directory):

```bash
vllm serve /path/to/dpo-strict-fp8 \
  --served-model-name struct-lm \
  --tokenizer-mode mistral \
  --limit-mm-per-prompt '{"image": 0}' \
  --config-format hf \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --seed 0
```

| flag | why |
|---|---|
| `--tokenizer-mode mistral` | Tekken through mistral-common, the tokenizer and chat rendering the model was trained and evaluated with |
| `--limit-mm-per-prompt '{"image": 0}'` | text only: the vision tower is never loaded and its memory goes to the KV cache |
| `--config-format hf` | the HF config and weights, the implementation every eval ran; the hub repo's Mistral-native `params.json` path is a different implementation |
| `--max-model-len 8192` | the eval's setting; the longest benchmark prompt is 2,094 tokens plus a 300-token answer cap |
| `--max-num-seqs 64` | the highest concurrency measured |
| `--enable-prefix-caching` | shared instruction prefixes and repeated retrieval contexts skip prefill (measured below) |

The precision variants differ only in the checkpoint directory (the quantization is read from each
checkpoint's `quantization_config`, compressed-tensors) and, for the FP8 KV cache, which failed the
gate, `KV_CACHE_DTYPE=fp8` (`--kv-cache-dtype fp8`).

## Request contract

- **`/v1/chat/completions`, one user message, no system prompt.** The server renders
  `<s>[INST]…[/INST]` with mistral-common, exactly the ids the trainer used (one BOS, one `[INST]`;
  `tests/test_template.py` checks it through the served endpoint for every variant). Answers end on
  `</s>`.
- **Never send pre-templated text**, and never `/v1/completions` with a hand-built template: the
  control tokens then arrive as ordinary text (the lm-eval bug, `notes/decisions.md`).
- **The trained formats:** closed-book ("Answer with the value, term or name only."), grounded
  (instructions, then `[chunk_id]` passages, then `Question: … Answer:`), and the abstain sentence
  "Not in the provided passages." for questions the passages don't answer.
- **Decoding:** greedy (the eval's setting). `max_tokens` 64 closed-book, 300 grounded, 200 abstain;
  the model stops itself well before these.
- **Retrieval layout:** put the passages before the question and keep the passage block
  byte-identical across questions that share it, so the prefix cache can reuse it.
- **Greedy is not bit-reproducible** across server starts or hosts, and isn't guaranteed under
  different batch compositions either: vLLM's kernels vary with batch shape. One of 20 smoke
  answers flipped at a 0.125-nat near-tie between two starts on different hosts. The eval numbers
  already include this: each checkpoint was generated in its own container, and the seed floors
  were measured the same way.

## The YaRN key

Every served `config.json` must carry `text_config.rope_parameters.apply_yarn_scaling: false`
(`train/merge.py`, `serve/quantize.py`). Without it vLLM 0.29 applies YaRN attention scaling the
model doesn't use: perplexity on the val slice reads +4.8% (7.23 against 6.89) and every eval number
degrades (3 MMLU points on the base). Check a new serving path with `eval/vllm_ppl.py`:

| checkpoint | vLLM val-slice perplexity | transformers (`perplexity.py`) |
|---|---|---|
| bf16 | 6.8945 | 6.8945 |
| FP8 | 6.9171 (+0.33%) | |

## Memory budget

KV cache per token = 2 (K, V) × 34 layers × 8 KV heads × 128 head dim × 2 bytes = 139,264 bytes
(136 KiB; vLLM's own accounting gives the same), so one 8,192-token sequence holds 1.14 GB, half
that with an FP8 KV cache. GQA (8 KV heads for 32 query heads) is what keeps it this small.

<!-- deploy-memory:start -->
| variant | GPU | weights (vLLM, GiB) | KV cache (GiB) | KV cache tokens | 8,192-token sequences: vLLM / Step 0 estimate |
|---|---|---|---|---|---|
| bf16 | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 15.94 | 52.75 | 406,688 | 49.6 / 44 |
| FP8 | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 9.01 | 60.61 | 467,296 | 57.0 / 50 |
| FP8 + FP8 KV | NVIDIA H100 80GB HBM3 (driver 610.57.04, vLLM 0.29.0) | 9.01 | 60.61 | 934,608 | 114.1 / 100 |
<!-- deploy-memory:end -->

## Expected latency (H100, measured)

`unique`: 360 eval requests (216 closed-book, 108 grounded, 36 abstain), greedy, each at its task's
token cap, prefix cache reset before each run, two runs per row. FP8 against bf16:
- **Faster decode:** 4.70 against 6.90 ms per token at one request.
- **Higher throughput:** +22% requests/s at 64 concurrent requests, +12% goodput at 32.
- **Slower first token at low load:** 35 against 18 ms p50 at one request, a near-constant
  10-20 ms per prefill (unexplained); by 64 requests it is lost in queueing.

The SLO used for goodput is TTFT <= 500 ms and TPOT <= 25 ms; both variants serve the most requests
inside it at 32 concurrent requests, and past that the queue breaks it for most requests.

<!-- deploy-latency:start -->
| variant | concurrency | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) | E2EL p50 / p99 (ms) | req/s | output tok/s | goodput share | run pair spread |
|---|---|---|---|---|---|---|---|---|
| bf16 | 1 | 17.9 / 54.0 | 6.90 / 7.80 | 71 / 964 | 4.78 | 130 | 100% | 0% |
| bf16 | 8 | 44.9 / 224.1 | 7.83 / 36.74 | 111 / 1,512 | 23.88 | 655 | 100% | 1% |
| bf16 | 32 | 87.9 / 638.4 | 10.90 / 88.83 | 286 / 3,058 | 40.28 | 1,096 | 70% | 0% |
| bf16 | 64 | 213.4 / 952.5 | 17.54 / 197.57 | 635 / 5,411 | 43.67 | 1,189 | 24% | 0% |
| FP8 | 1 | 35.2 / 59.9 | 4.70 / 5.63 | 71 / 804 | 5.94 | 164 | 100% | 0% |
| FP8 | 8 | 79.8 / 214.6 | 5.30 / 49.55 | 149 / 1,688 | 21.20 | 580 | 99% | 2% |
| FP8 | 32 | 128.3 / 478.6 | 7.66 / 59.36 | 290 / 3,172 | 40.75 | 1,117 | 77% | 4% |
| FP8 | 64 | 222.5 / 717.2 | 10.53 / 127.30 | 575 / 4,434 | 53.08 | 1,461 | 30% | 1% |
<!-- deploy-latency:end -->

**Plausibility floor.** Decoding one request is memory-bandwidth-bound: every step reads the
linear weights and `lm_head` once. At the H100's 3.35 TB/s that is at least 4.75 ms per token for
bf16 (15.9 GB), 2.5 ms for FP8 (8.5 GB) and 1.5 ms for INT4 (4.9 GB, `lm_head` still bf16). A bf16
inter-token latency under 4.75 ms at concurrency 1 did not run on an H100 (an H200 at 4.8 TB/s sits
30% lower), whatever the request said.

## Quality gate

<!-- deploy-gate:start -->
| line | bf16 | FP8 − bf16 | FP8 + FP8 KV − bf16 | INT4 W4A16 − bf16 | floor |
|---|---|---|---|---|---|
| qa_strict unseen | 0.110 | -1.3 | +0.6 | -2.6 | 2.6 pt |
| qa_strict seen | 0.305 | -1.2 | **-4.2** | -3.0 | 3.5 pt |
| gold_lp answer tokens, unseen (nats) | -6.274 | +0.000 | -0.055 | **-0.424** | 0.210 |
| gold_lp answer tokens, seen (nats) | -4.811 | -0.007 | -0.056 | **-0.307** | 0.173 |
| gold_lp end token, unseen (nats) | -0.421 | -0.020 | -0.029 | **+0.059** | 0.048 |
| gold_lp end token, seen (nats) | -0.319 | -0.004 | -0.008 | +0.015 | 0.074 |
| grounded_acc | 0.926 | -1.8 | +0.9 | -2.8 | 2.8 pt |
| cite_valid | 1.000 | +0.0 | +0.0 | +0.0 | 1.85 pt |
| halluc_rate | 0.040 | +1.3 | +2.6 | +0.0 | 3.9 pt |
| false_abstain | 0.000 | +0.0 | +0.0 | +0.0 | 0.9 pt |
| GSM8K (all 1,319, add_bos_token) | 0.809 | -0.2 |  | **-6.1** | 2.2 pt |
| qa_strict identifiers (reported) | 0.188 | -1.6 | -1.6 | **-7.8** | 5 pt |
| answers ending on </s> (eos job) | 1.00 | 1.00 | 1.00 | 1.00 | >= 0.95 |
| vLLM val-slice perplexity (reported) | 6.894 | +0.33% | +0.58% | +3.53% |  |
| **verdict** |  | **ships** | **fails** | **fails** |  |
<!-- deploy-gate:end -->

## Cost

**Where the 8B stops being the right choice.**
- **The GPU only pays at sustained load.** One H100 (bf16 or FP8) beats Mistral Small 4's API price
  only above ~9 requests/s sustained (31k an hour, at this request mix). Below that, the API is
  cheaper per request, because an idle GPU still bills by the hour.
- **The 8B's case is the customer's facts on one GPU.** Self-hosting Small 4 needs at least four
  H100s.
- **General capability is where it stops.** There, a 119B MoE like Small 4 is the likelier winner;
  that comparison wasn't run here.

<!-- deploy-cost:start -->
| variant | gate | best concurrency | goodput (req/s) | $ / 1k requests (1 H100) | API $ / 1k requests (Small 4 prices) | break-even sustained req/s |
|---|---|---|---|---|---|---|
| bf16 | reference | 32 | 28.1 | 0.0390 | 0.1266 | 8.7 |
| FP8 | ships | 32 | 31.5 | 0.0348 | 0.1267 | 8.7 |

One H100 at $3.95/h (Modal's H100 SXM5 list price, $0.001097/s, checked 2026-10-09); Small 4's API at $0.15 / $0.6 per million input / output tokens and its self-hosting minimum of 4 H100s ($15.80/h before any request), both from Mistral's Small 4 announcement. Tokens per request are the bench mix's measured means. The break-even is the sustained load above which the GPU is cheaper than the API; below it, an idle GPU costs the same per hour.
<!-- deploy-cost:end -->
