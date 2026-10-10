# struct-lm

> Independent project, using only public documents, open weights and open-source tools. Forge is
> described from Mistral AI's public announcement.

Domain adaptation of an open base LLM
([`mistralai/Ministral-3-8B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512))
to structural and civil engineering through **CPT → SFT → DPO → GRPO**, with every stage
measured on the same KPI tasks and general-capability regression suite, then quantized
(FP8, INT4) behind a pre-registered quality gate and served with vLLM on one H100
([`DEPLOY.md`](DEPLOY.md)).

> This README is the write-up. Numbers live in [`results/table.md`](results/table.md);
> the reasoning behind every choice lives in [`notes/decisions.md`](notes/decisions.md).

## Contents

- [Stage 0: the problem, the eval and the baselines](docs/stage0.md)
- [Stage 1: the corpus](docs/stage1.md)
- [Stage 2: continued pre-training (CPT)](docs/stage2.md)
- [Stage 3: supervised fine-tuning (SFT)](docs/stage3.md)
- [Stage 4: DPO on verifiable preferences](docs/stage4.md)
- [Stage 5: GRPO with verifiable rewards](docs/stage5.md)
- [Stage 6: serving](docs/stage6.md)
- [Results: every scored checkpoint](docs/results.md)
- [Reproduce: pipeline, quickstart, layout](docs/reproduce.md)
