"""Run the KPI eval and the lm-eval regression suite on a Modal H100.

  modal run eval/modal_app.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b
  modal run eval/modal_app.py --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --run-name instruct-8b --chat
  modal run eval/modal_app.py --model ... --run-name smoke --limit 5 --no-judge --which kpi
  modal run eval/modal_app.py --model mistralai/Ministral-3-8B-Base-2512 --run-name base-8b --which latency
  modal volume get struct-lm results .        # pull results/ back into the repo

To keep judge calls off the GPU clock, generate on Modal with --generate-only, pull results/,
then score everything locally: python eval/run_eval.py --run-name <r> --rescore
--lm-eval-dir results/lm_eval

Secrets expected in Modal: `huggingface` (HF_TOKEN) and `mistral` (MISTRAL_API_KEY).
The volume `struct-lm` holds the HF cache (so the weights download once), results,
and later the checkpoints.

How it fits together: `modal run` executes main() on your machine. main() calls the two GPU
functions with .remote(), which blocks until each finishes in its own container. Each container
runs the same scripts you would run by hand (run_lm_eval.sh, run_eval.py) as subprocesses,
reading and writing /vol, then commits the volume so the results persist.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

app = modal.App("struct-lm-eval")
# Persistent storage shared by every run, mounted at /vol in both functions:
#   /vol/hf       Hugging Face cache (HF_HOME below): each checkpoint downloads once
#   /vol/results  same layout as the repo's results/, pulled with `modal volume get`
vol = modal.Volume.from_name("struct-lm", create_if_missing=True)

# The container image, built once and cached by Modal until this definition changes. Changing
# a pin here triggers a rebuild on the next `modal run`.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.29.0",
        # 5.17 renamed PixtralRotaryEmbedding -> PixtralVisionRotaryEmbedding; vLLM 0.29/0.30
        # still import the old name, so Ministral 3 (Pixtral vision tower) fails to load.
        "transformers>=5.10.4,<5.17",
        "mistral-common>=1.8.6",  # Tekken tokenizer for tokenizer_mode="mistral"
        "mistralai>=2.0",  # judge.py imports mistralai.client (2.x layout)
        "mistral_inference",  # needs xformers CUDA kernels, so it lives here, not on the Mac
        "lm_eval[vllm]>=0.4.13",  # handles vLLM's MistralTokenizer (tokenizer_mode=mistral)
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            # huggingface_hub 1.x (required by transformers 5) dropped hf_transfer for Xet
            "HF_XET_HIGH_PERFORMANCE": "1",
            # vLLM workers must be spawned, not forked: forking a process that has already
            # initialised CUDA fails.
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # FlashInfer JIT-compiles its top-k/top-p sampler with nvcc, which debian_slim lacks
            # (engine dies in warmup: "Could not find nvcc"). Use vLLM's PyTorch sampler instead;
            # eval decoding is greedy, so outputs are unaffected.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    # Your local eval/ directory, copied in at every launch rather than baked into the image, so
    # edits to tasks, prompts or scorers take effect without a rebuild. It also means the run
    # evaluates whatever is in eval/tasks/ right now: don't regenerate tasks between stages.
    .add_local_dir("eval", remote_path="/root/eval")  # tasks, prompts, scorers, judge, run_eval
    .add_local_dir("serve", remote_path="/root/serve")  # serve_vllm.sh, bench_latency.py
    .add_local_dir("train", remote_path="/root/train")  # packing.py, for vllm_ppl.py's windows
)

# Settings shared by all GPU functions. 4 h covers a full 8B lm-eval run (MMLU is the long
# pole) with margin; the secrets become environment variables inside the container
# (HF_TOKEN for gated weights, MISTRAL_API_KEY for the judge).
COMMON = {
    "image": image,
    "gpu": "H100",
    "timeout": 4 * 3600,
    "volumes": {"/vol": vol},
    "secrets": [modal.Secret.from_name("huggingface"), modal.Secret.from_name("mistral")],
}


@app.function(**COMMON)
def kpi_eval(
    model: str,
    run_name: str,
    chat: bool,
    limit: int | None,
    no_judge: bool,
    tokenizer_mode: str,
    generate_only: bool = False,
    tasks: str = "",
    config_format: str = "hf",
    gold_lp_only: bool = False,
) -> None:
    """The domain KPI eval: run_eval.py with container paths. Arguments map 1:1 to its flags.
    `tasks` (comma-separated, "" = all) regenerates only those tasks; the run's other saved
    generations on the volume are kept, so push the local generations.jsonl first.

    Writes /vol/results/runs/<run_name>/ and, unless generate_only, a row in
    /vol/results/table.md. --lm-eval-dir points at lm_eval()'s output, so if that ran first
    under the same run_name, its MMLU / GSM8K / HellaSwag numbers land in the same row."""
    cmd = [
        sys.executable,
        "/root/eval/run_eval.py",
        "--model",
        model,
        "--run-name",
        run_name,
        "--tasks-dir",
        "/root/eval/tasks",
        "--results-dir",
        "/vol/results",
        "--lm-eval-dir",
        "/vol/results/lm_eval",
        "--tokenizer-mode",
        tokenizer_mode,
    ]
    if chat:
        cmd.append("--chat")
    if limit:
        cmd += ["--limit", str(limit)]
    if no_judge:
        cmd.append("--no-judge")
    if generate_only:
        cmd.append("--generate-only")
    if tasks:
        cmd += ["--tasks", tasks]
    cmd += ["--config-format", config_format]
    if gold_lp_only:
        cmd.append("--gold-lp-only")
    subprocess.run(cmd, check=True, cwd="/root")  # check=True: a failed eval fails the Modal call
    vol.commit()  # persist results; without this, writes to /vol are lost when the container exits


@app.function(**COMMON)
def lm_eval(model: str, run_name: str, tokenizer_mode: str) -> None:
    """The general-capability regression suite: run_lm_eval.sh, writing
    /vol/results/lm_eval/<run_name>/. Always without the chat template (see run_lm_eval.sh).
    The script takes the tokenizer mode as an environment variable, so it goes through env."""
    subprocess.run(
        ["bash", "/root/eval/run_lm_eval.sh", model, run_name, "/vol/results"],
        check=True,
        cwd="/root",
        env={**os.environ, "TOKENIZER_MODE": tokenizer_mode},
    )
    vol.commit()


@app.function(**COMMON)
def latency(model: str, run_name: str, tokenizer_mode: str) -> None:
    """Serving benchmark: start serve_vllm.sh (the OpenAI-compatible server used in production)
    in the background, wait until it answers /health, run bench_latency.py against it, and
    write /vol/results/bench/<run_name>.json (TTFT / ITL / E2E p50-p95 and tok/s per
    concurrency level). The server is stopped however the benchmark ends.

    The benchmark always goes through the chat endpoint, so there's no `chat` switch: this
    measures serving speed, not answer quality. MAX_MODEL_LEN is 8192, the same as run_eval.py,
    so any prompt the KPI eval can send (4 passages plus instructions, plus 256 output tokens)
    also fits here."""
    server = subprocess.Popen(
        ["bash", "/root/serve/serve_vllm.sh", model],
        cwd="/root",
        env={**os.environ, "TOKENIZER_MODE": tokenizer_mode, "MAX_MODEL_LEN": "8192"},
    )
    try:
        # Loading 8B weights from the volume and compiling takes a few minutes; a first-time
        # download takes longer. Fail fast if the server process dies instead of waiting.
        deadline = time.monotonic() + 30 * 60
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {server.returncode} before ready")
            try:
                with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:  # connection refused / not ready yet (URLError is an OSError)
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("vLLM server not ready after 30 min")
            time.sleep(5)
        subprocess.run(
            [
                sys.executable,
                "/root/serve/bench_latency.py",
                "--label",
                run_name,
                "--tasks",
                "/root/eval/tasks",
                "--out-dir",
                "/vol/results/bench",
            ],
            check=True,
            cwd="/root",
        )
    finally:
        server.terminate()
        server.wait(timeout=120)
    vol.commit()


@app.function(**COMMON)
def vllm_ppl(model: str, config_format: str = "hf", no_yarn_scale: bool = False) -> None:
    """eval/vllm_ppl.py: vLLM's perplexity on the trainer's val slice, to compare with transformers'
    ppl_val_slice. Writes /vol/results/vllm_ppl/<model>-<format>[-noyarn].json.
      modal run eval/modal_app.py::vllm_ppl --model /vol/checkpoints/base-8b-hf --config-format hf"""
    vol.reload()
    name = f"{Path(model).name}-{config_format}" + ("-noyarn" if no_yarn_scale else "")
    cmd = [
        sys.executable,
        "/root/eval/vllm_ppl.py",
        "--model",
        model,
        "--config-format",
        config_format,
        "--out",
        f"/vol/results/vllm_ppl/{name}.json",
    ]
    subprocess.run(cmd + (["--no-yarn-scale"] if no_yarn_scale else []), check=True, cwd="/vol")
    vol.commit()


@app.function(**COMMON)
def memorization(model: str, run_name: str, config_format: str = "hf") -> None:
    """eval/memorization.py run: verbatim recall and per-document perplexity for the train, val
    and post-cutoff documents in /vol/data/exposure/docs.jsonl. Writes
    /vol/results/exposure/<run_name>.json.
      modal run eval/modal_app.py::memorization --model /vol/checkpoints/base-8b-hf --run-name base-8b-hf"""
    vol.reload()
    cmd = [
        sys.executable,
        "/root/eval/memorization.py",
        "run",
        "--model",
        model,
        "--run-name",
        run_name,
        "--config-format",
        config_format,
    ]
    subprocess.run(cmd, check=True, cwd="/vol")
    vol.commit()


WHICH = ("lm", "kpi", "both", "latency")


@app.local_entrypoint()
def main(
    model: str,
    run_name: str,
    chat: bool = False,
    limit: int = 0,
    no_judge: bool = False,
    generate_only: bool = False,
    which: str = "both",
    tokenizer_mode: str = "mistral",  # "auto" for non-Mistral-3 checkpoints
    tasks: str = "",  # KPI only: e.g. "domain_qa" regenerates that task and keeps the others
    config_format: str = "hf",  # KPI only: "auto" to extend Stage 0's native-path hub runs
    gold_lp_only: bool = False,  # KPI only: recompute gold_lp in the saved generations, no generation
) -> None:
    """Runs locally. Modal turns each parameter into a CLI flag (run_name -> --run-name,
    bools -> --chat / --no-chat). `which` picks "lm", "kpi", "both" (lm then kpi) or
    "latency" (serving benchmark only; it ignores chat, limit, no_judge and generate_only).

    `chat` applies to the KPI eval only: chat checkpoints always run it with their chat
    template, and lm-eval never uses one (see run_lm_eval.sh). `limit` 0 means all items
    (Modal flags can't be None)."""
    if which not in WHICH:  # an unknown value would otherwise match no branch and do nothing
        raise SystemExit(f"--which must be one of {', '.join(WHICH)}, got {which!r}")
    if which in ("kpi", "both") and "instruct" in model.lower() and not chat:
        # Checked here so the mistake fails locally, before a GPU container starts.
        raise SystemExit("KPI eval of an Instruct checkpoint must use --chat")
    if which in ("lm", "both"):
        lm_eval.remote(model, run_name, tokenizer_mode)  # first, so kpi_eval can merge its numbers
    if which in ("kpi", "both"):
        kpi_eval.remote(
            model,
            run_name,
            chat,
            limit or None,
            no_judge,
            tokenizer_mode,
            generate_only,
            tasks,
            config_format,
            gold_lp_only,
        )
    if which == "latency":
        latency.remote(model, run_name, tokenizer_mode)
