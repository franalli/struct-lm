"""Stage 6 on Modal: quantize stage5-final (checkpoints/dpo-strict) and benchmark every precision
variant in one container on one pinned H100.

  M=.venv/bin/modal
  $M run serve/modal_serve.py --action quantize --scheme fp8           # -> /vol/checkpoints/dpo-strict-fp8
  $M run --detach serve/modal_serve.py --action quantize --scheme w4a16
  $M run serve/modal_serve.py --action bench --variants bf16 --check-only
  $M run --detach serve/modal_serve.py --action bench --variants bf16,fp8,fp8kv,w4a16 --spec
  $M volume get --force struct-lm results/serve/bench results/serve/   # then served checks, logs

quantize: serve/quantize.py in its own image (llmcompressor 0.14.0 and compressed-tensors 0.19; vLLM
0.29 pins compressed-tensors 0.17, so they can't share one). The quality gate's generations run
through eval/modal_app.py as for any checkpoint (--kv-cache-dtype fp8 for fp8kv).

bench: the eval image (vLLM 0.29.0, the eval's own pins), gpu "H100!" (Modal may otherwise run an
"H100" request on an H200, whose ~40% more memory bandwidth would move every decode number), and a
runtime check that refuses anything but an H100: device name, compute capability, driver, CUDA, vLLM
and torch go into every result. Variants run in sequence in the one container, the server restarted
for each, so the hardware is identical by construction. Per variant: the server's own memory lines
(weights, KV cache tokens, max concurrency at 8,192 tokens), serve/served_check.py, then `vllm bench
serve` on serve/bench_data.py's sets (rebuilt here and checked against serve/bench_manifest.json):
concurrency 1 / 8 / 32 / 64 and Poisson 1 / 4 / 16 req/s on `unique`, each run twice; fp8 adds
grounded_unique vs grounded_rag at 8 (the prefix cache) and the closed-book / grounded runs the
speculative-decoding rows compare with. Before every run the prefix cache is reset (the server runs
with VLLM_SERVER_DEV_MODE=1 here only, never in DEPLOY.md), and the run's prefix-cache hit rate from
/metrics is recorded, which shows the reset worked. Results are compact per-run JSON under
/vol/results/serve/bench/<variant>/: vLLM's summary, per-request ttft / tpot / e2el / lengths with
each request's task, goodput (TTFT <= 500 ms and TPOT <= 25 ms), committed after every variant.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "modal_app.py").exists():
        sys.path.insert(0, str(_d))
        break
from modal_app import app as eval_app
from modal_app import image as eval_image

app = modal.App("struct-lm-serve").include(eval_app)
vol = modal.Volume.from_name("struct-lm", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")]

# The trainer's pins (train/modal_train.py) plus llm-compressor: the model loads exactly as merge.py
# loaded it. No vLLM here.
quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "llmcompressor==0.14.0",  # pins compressed-tensors 0.19.0
        "peft==0.21.0",  # train/common.py imports it
        "datasets==5.0.1",
        "mistral-common==1.12.0",
        "numpy",
        "pyyaml",
    )
    .env({"HF_HOME": "/vol/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("train", remote_path="/root/train")
    .add_local_dir("eval", remote_path="/root/eval")  # gold_lp.py, which sft_data imports
    .add_local_dir("serve", remote_path="/root/serve")
)

SOURCE = "/vol/checkpoints/dpo-strict"  # stage5-final (notes/decisions.md, 2026-10-09)
CKPT = {
    "bf16": SOURCE,
    "fp8": f"{SOURCE}-fp8",
    "w4a16": f"{SOURCE}-w4a16",
}
VARIANTS = {
    "bf16": {"model": CKPT["bf16"], "env": {}},
    "fp8": {"model": CKPT["fp8"], "env": {}},
    "fp8kv": {"model": CKPT["fp8"], "env": {"KV_CACHE_DTYPE": "fp8"}},
    "w4a16": {"model": CKPT["w4a16"], "env": {}},
    # the pasted plan's fallback if the offline FP8 save can't load: vLLM's online dynamic FP8
    "fp8-online": {"model": CKPT["bf16"], "env": {"QUANTIZATION": "fp8"}},
}
SPEC = '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
CONCURRENCY = (1, 8, 32, 64)
RATES = (1, 4, 16)
GOODPUT = {"ttft": 500, "tpot": 25}  # ms: the stated SLO
BENCH_DIR = Path("/tmp/bench")
OUT = Path("/vol/results/serve")


@app.function(
    image=quant_image, gpu="H100", timeout=3 * 3600, volumes={"/vol": vol}, secrets=SECRETS
)
def quantize(scheme: str) -> None:
    vol.reload()
    cmd = [
        sys.executable,
        "/root/serve/quantize.py",
        "--model",
        SOURCE,
        "--scheme",
        scheme,
        "--out",
        CKPT[scheme],
        "--calib",
        "data/sft/train.jsonl",  # on the volume (CLAUDE.md, Stage 3 data upload)
    ]
    subprocess.run(cmd, check=True, cwd="/vol")
    vol.commit()


# ------------------------------------------------------------------- bench ---
def gpu_info() -> dict:
    """From nvidia-smi, so this process never opens a CUDA context next to the server's."""
    import torch
    import vllm

    q = "name,compute_cap,driver_version,memory.total"
    line = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    name, cc, driver, mem = (s.strip() for s in line.split(","))
    info = {
        "gpu": name,
        "compute_capability": cc,
        "driver": driver,
        "memory": mem,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
    }
    if "H100" not in name:
        raise SystemExit(f"not an H100: {info}; no result written")
    return info


def start_server(model: str, env: dict, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        ["bash", "/root/serve/serve_vllm.sh", model],
        cwd="/root",
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1", **env},
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30 * 60
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode} before ready; see {log}")
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                if r.status == 200:
                    return server
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("vLLM server not ready after 30 min")
        time.sleep(5)


def stop_server(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


SERVER_LINES = {
    "weights_gib": r"Model loading took ([\d.]+) GiB",
    "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency_8192": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "quantization": r"quantization=([\w-]+)",
    "kv_cache_dtype": r"kv_cache_dtype=([\w-]+)",
    # e.g. "Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8"
    "linear_kernel": r"Selected (\w+ for CompressedTensors\w+)",
}


def server_stats(log: Path) -> dict:
    """vLLM's own memory accounting (the Step 0 budget's check) and the loaded config's flags."""
    text = log.read_text(errors="replace")
    out = {}
    for k, pat in SERVER_LINES.items():
        m = re.findall(pat, text)
        out[k] = m[-1].replace(",", "") if m else None
    for k in ("weights_gib", "kv_cache_gib", "max_concurrency_8192"):
        out[k] = float(out[k]) if out[k] else None
    out["kv_cache_tokens"] = int(out["kv_cache_tokens"]) if out["kv_cache_tokens"] else None
    return out


def post(path: str) -> int:
    req = urllib.request.Request(f"http://localhost:8000{path}", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def warm_up(n: int = 8) -> None:
    """A few short requests outside every bench set, sent
    before the prefix-cache reset, so the measured run starts warm but with an empty cache."""
    for k in range(n):
        body = {
            "model": "struct-lm",
            "messages": [{"role": "user", "content": f"Define warm-up term {k} in one sentence."}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()


def prefix_counters() -> dict:
    """vllm:prefix_cache_queries / _hits (tokens), summed over label sets; {} if not exported."""
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    pat = r"^vllm:prefix_cache_(hits|queries)(?:_total)?(?:\{[^}]*\})?\s+(\S+)$"
    for m in re.finditer(pat, text, re.MULTILINE):
        out[m[1]] = out.get(m[1], 0.0) + float(m[2])
    return out


def compact(raw: dict, rows: list[dict]) -> dict:
    """vLLM's summary plus per-request arrays small enough to commit (no texts, no per-token
    ITLs): each request's task, ttft, tpot, e2el, lengths, and whether it hit its cap."""
    if len(raw["ttfts"]) != len(rows):
        raise SystemExit(f"{len(raw['ttfts'])} results for {len(rows)} requests")
    keep = {k: v for k, v in raw.items() if not isinstance(v, list)}
    req = []
    for r, ttft, itls, n_in, n_out, err in zip(
        rows, raw["ttfts"], raw["itls"], raw["input_lens"], raw["output_lens"], raw["errors"]
    ):
        e2el = ttft + sum(itls)
        req.append(
            {
                "task": r["task"],
                "id": r["id"],
                "ttft_ms": round(ttft * 1000, 2),
                "tpot_ms": round((e2el - ttft) / (n_out - 1) * 1000, 3) if n_out > 1 else None,
                "e2el_ms": round(e2el * 1000, 2),
                "in": n_in,
                "out": n_out,
                "hit_cap": n_out >= r["output_tokens"],
                "error": err or None,
            }
        )
    ok = [q for q in req if not q["error"]]
    good = [
        q
        for q in ok
        if q["ttft_ms"] <= GOODPUT["ttft"] and (q["tpot_ms"] or 0.0) <= GOODPUT["tpot"]
    ]
    keep["goodput_share"] = round(len(good) / len(ok), 4) if ok else None
    keep["requests"] = req
    return keep


def bench_run(
    model: str,
    variant: str,
    dataset: str,
    name: str,
    seed: int,
    out_dir: Path,
    meta: dict,
    concurrency: int | None = None,
    rate: float | str = "inf",
    limit: int | None = None,
) -> None:
    rows = [json.loads(line) for line in (BENCH_DIR / f"{dataset}.jsonl").read_text().splitlines()]
    path = BENCH_DIR / f"{dataset}.jsonl"
    if limit:  # the check-only smoke: the head of the file
        rows = rows[:limit]
        path = BENCH_DIR / f"{dataset}-head{limit}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    warm_up()
    assert post("/reset_prefix_cache") == 200
    before = prefix_counters()
    raw_dir = Path("/vol/serve_raw") / variant
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--endpoint", "/v1/chat/completions",
        "--model", model,
        "--served-model-name", "struct-lm",
        "--tokenizer-mode", "mistral",
        "--dataset-name", "custom",
        "--dataset-path", str(path),
        "--skip-chat-template",  # the server renders the template; never twice (rule 2)
        "--custom-output-len", "-1",  # each row's output_tokens: the task's eval cap
        "--num-prompts", str(len(rows)),
        "--no-oversample",  # a repeated prompt would hit the prefix cache
        "--disable-shuffle",  # file order: grounded_unique and grounded_rag ask in the same order
        "--temperature", "0",
        # vLLM's own warm-up repeats the set's first prompt, which would then be cached for the
        # measured run; warm_up() runs before the reset instead
        "--num-warmups", "0",
        "--request-rate", str(rate),
        "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--goodput", *(f"{k}:{v}" for k, v in GOODPUT.items()),
        "--save-result", "--save-detailed",
        "--result-dir", str(raw_dir),
        "--result-filename", f"{name}.json",
        "--label", f"{variant}-{name}",
    ]  # fmt: skip
    if concurrency:
        cmd += ["--max-concurrency", str(concurrency)]
    subprocess.run(cmd, check=True, cwd="/root")
    after = prefix_counters()
    raw = json.loads((raw_dir / f"{name}.json").read_text())
    res = compact(raw, rows)
    q = after.get("queries", 0.0) - before.get("queries", 0.0)
    hits = after.get("hits", 0.0) - before.get("hits", 0.0)
    res["prefix_cache"] = {
        "queries": q,
        "hits": hits,
        "hit_rate": round(hits / q, 4) if q else None,
    }
    res["run"] = {"variant": variant, "dataset": dataset, "name": name, "seed": seed,
                  "max_concurrency": concurrency, "request_rate": rate, **meta}  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(
        f"{variant} {name}: {res.get('request_throughput', 0):.2f} req/s, goodput share "
        f"{res['goodput_share']}, prefix hit rate {res['prefix_cache']['hit_rate']}"
    )


def spec_run(dataset: str, name: str, concurrency: int | None = None, rate="inf") -> dict:
    return {"dataset": dataset, "name": name, "concurrency": concurrency, "rate": rate}


def runs_for(variant: str, repeats: int) -> list[dict]:
    plan = [spec_run("unique", f"c{c}", concurrency=c) for c in CONCURRENCY]
    plan += [spec_run("unique", f"r{r}", rate=r) for r in RATES]
    if variant == "fp8":  # the prefix-cache pair and the speculative-decoding baselines
        plan += [
            spec_run("grounded_unique", "grounded-unique-c8", concurrency=8),
            spec_run("grounded_rag", "grounded-rag-c8", concurrency=8),
            spec_run("grounded_unique", "grounded-unique-c1", concurrency=1),
            spec_run("closedbook", "closedbook-c1", concurrency=1),
            spec_run("closedbook", "closedbook-c8", concurrency=8),
        ]
    out = []
    for rep in range(repeats):
        out += [{**p, "name": f"{p['name']}-rep{rep}"} for p in plan]
    return out


def serve_variant(
    variant: str,
    model: str,
    env: dict,
    info: dict,
    check_only: bool,
    repeats: int,
    runs: list[dict] | None = None,
) -> None:
    out_dir = OUT / "bench" / variant
    log = Path("/vol/serve_logs") / f"{variant}.log"
    server = start_server(model, env, log)
    try:
        meta = {"model": model, "server_env": env, **info, "server": server_stats(log)}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "server.json").write_text(json.dumps(meta, indent=2) + "\n")
        # one file per server start: the check-only launch and the bench's own start both keep theirs
        tag = "served_check" if check_only else "served_check_bench"
        check = [sys.executable, "/root/serve/served_check.py", "--variant", variant,
                 "--out", str(OUT / variant / f"{tag}.jsonl")]  # fmt: skip
        if variant == "bf16":  # the 20-prompt identity smoke: the eval generated from these weights
            check += ["--smoke", "/vol/results/runs/dpo-strict/generations.jsonl"]
        subprocess.run(check, check=True, cwd="/root")
        if check_only:  # plus one small bench run: the invocation itself works
            bench_run(model, variant, "unique", "smoke-c8", 0, out_dir / "smoke", meta,
                      concurrency=8, limit=24)  # fmt: skip
        else:
            for k, r in enumerate(runs or runs_for(variant, repeats)):
                bench_run(model, variant, r["dataset"], r["name"], 1000 + k, out_dir, meta,
                          concurrency=r["concurrency"], rate=r["rate"])  # fmt: skip
    finally:
        stop_server(server)
        vol.commit()


@app.function(
    image=eval_image, gpu="H100!", timeout=6 * 3600, volumes={"/vol": vol}, secrets=SECRETS
)
def bench(variants: str, check_only: bool = False, spec: bool = False, repeats: int = 2) -> None:
    vol.reload()
    info = gpu_info()  # refuses anything but an H100 before a server starts
    print(info)
    subprocess.run(
        [sys.executable, "/root/serve/bench_data.py", "--check", "--out-dir", str(BENCH_DIR)],
        check=True,
        cwd="/root",
    )
    for v in [v for v in variants.split(",") if v]:
        serve_variant(v, VARIANTS[v]["model"], VARIANTS[v]["env"], info, check_only, repeats)
    if spec and not check_only:  # step 6: n-gram speculative decoding on the FP8 checkpoint
        runs = [
            spec_run(d, f"{d}-c{c}-rep{rep}", concurrency=c)
            for rep in range(repeats)
            for d in ("grounded_unique", "closedbook")
            for c in (1, 8)
        ]
        serve_variant("fp8-ngram", CKPT["fp8"], {"SPEC_CONFIG": SPEC}, info, False, repeats, runs)


# -------------------------------------------------------------------- gate ---
GATE_RUN = {  # the run names the gate's generations are filed under (needs_chat sees "dpo")
    "bf16": "dpo-strict",
    "fp8": "dpo-strict-fp8",
    "fp8kv": "dpo-strict-fp8kv",
    "w4a16": "dpo-strict-w4a16",
}
GATE_STEPS = ("kpi", "eos", "ppl", "gsm8k")


@app.function(
    image=eval_image, gpu="H100", timeout=4 * 3600, volumes={"/vol": vol}, secrets=SECRETS
)
def gate(variant: str, steps: str = "") -> None:
    """Stage 6's pre-registered quality gate for one variant (notes/decisions.md), one container:
      kpi    run_eval.py --chat --generate-only (scored on the Mac against dpo-strict's own rows)
      eos    sample.py's eos job: the share of answers that end on </s>
      ppl    vllm_ppl.py on the trainer's val slice (reported; with bf16 it is the YaRN path check)
      gsm8k  eval/gsm8k_gate.py: GSM8K, all 1,319, add_bos_token=True, into results/serve/gate/
    bf16 runs ppl and gsm8k (its KPI rows and eos samples are dpo-strict's), plus the BOS probe."""
    vol.reload()
    run, model = GATE_RUN[variant], VARIANTS[variant]["model"]
    kv = VARIANTS[variant]["env"].get("KV_CACHE_DTYPE", "auto")
    todo = [s for s in steps.split(",") if s] or (
        ["ppl", "gsm8k"] if variant == "bf16" else list(GATE_STEPS)
    )
    if unknown := set(todo) - set(GATE_STEPS):
        raise SystemExit(f"unknown gate steps {sorted(unknown)}")
    py = sys.executable
    if "kpi" in todo:
        subprocess.run(
            [py, "/root/eval/run_eval.py", "--model", model, "--run-name", run, "--chat",
             "--generate-only", "--tasks-dir", "/root/eval/tasks", "--results-dir", "/vol/results",
             "--lm-eval-dir", "/vol/results/lm_eval", "--kv-cache-dtype", kv],
            check=True, cwd="/root",
        )  # fmt: skip
        vol.commit()
    if "eos" in todo:
        subprocess.run(
            [py, "/root/eval/sample.py", "--model", model, "--run-name", run, "--jobs", "eos",
             "--chat", "--results-dir", "/vol/results", "--kv-cache-dtype", kv],
            check=True, cwd="/vol",  # data/dpo/prompts.jsonl, the eos job's prompts
        )  # fmt: skip
        vol.commit()
    if "ppl" in todo:
        subprocess.run(
            [py, "/root/eval/vllm_ppl.py", "--model", model, "--config-format", "hf",
             "--kv-cache-dtype", kv, "--out", f"/vol/results/vllm_ppl/{run}-hf.json"],
            check=True, cwd="/vol",
        )  # fmt: skip
        vol.commit()
    if "gsm8k" in todo:
        if variant == "bf16":  # what add_bos_token changes in the ids lm-eval sends
            subprocess.run(
                [py, "/root/eval/bos_probe.py", "--model", model,
                 "--out", "/vol/results/serve/gate/bos_probe.json"],
                check=True, cwd="/root",
            )  # fmt: skip
        subprocess.run(
            [py, "/root/eval/gsm8k_gate.py", "--model", model, "--kv-cache-dtype", kv,
             "--out", f"/vol/results/serve/gate/lm_eval/{run}"],
            check=True, cwd="/root",
        )  # fmt: skip
        vol.commit()


@app.local_entrypoint()
def main(
    action: str,
    scheme: str = "",
    variants: str = "bf16",
    check_only: bool = False,
    spec: bool = False,
    repeats: int = 2,
    variant: str = "",
    steps: str = "",
) -> None:
    """--action quantize --scheme fp8|w4a16; --action gate --variant v [--steps kpi,eos,ppl,gsm8k];
    --action bench --variants a,b,... [--check-only] [--spec]. Variants: bf16, fp8, fp8kv, w4a16,
    fp8-online (the bench's fallback row)."""
    if action == "quantize":
        if scheme not in ("fp8", "w4a16"):
            raise SystemExit("--scheme fp8 or w4a16")
        quantize.remote(scheme)
    elif action == "bench":
        if unknown := set(variants.split(",")) - set(VARIANTS):
            raise SystemExit(f"unknown variants {sorted(unknown)}; choose from {list(VARIANTS)}")
        bench.remote(variants, check_only, spec, repeats)
    elif action == "gate":
        if variant not in GATE_RUN:
            raise SystemExit(f"--variant one of {list(GATE_RUN)}")
        gate.remote(variant, steps)
    else:
        raise SystemExit("--action quantize, gate or bench")
