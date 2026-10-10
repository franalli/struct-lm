"""The README's demo: the shipped FP8 checkpoint behind vLLM's OpenAI server on one H100, on an
ephemeral Modal URL that needs an API key.

  export DEMO_API_KEY=$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(24))')
  .venv/bin/modal serve serve/modal_demo.py    # prints the URL; the first start takes minutes
  .venv/bin/python serve/demo.py --url <URL>   # one grounded question, one unanswerable one

`modal serve` keeps the app up only while it runs (Ctrl-C stops it and the GPU). The server is
serve/serve_vllm.sh, DEPLOY.md's command, in the eval image whose pins the gate and bench used; the
key reaches vLLM as VLLM_API_KEY, so requests without it get 401.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

for _d in (Path(__file__).resolve().parents[1] / "eval", Path("/root/eval")):
    if (_d / "modal_app.py").exists():
        sys.path.insert(0, str(_d))
        break
from modal_app import image

CHECKPOINT = "/vol/checkpoints/dpo-strict-fp8"
app = modal.App("struct-lm-demo")
vol = modal.Volume.from_name("struct-lm")
key = os.environ.get("DEMO_API_KEY")  # read where `modal serve` runs, never written to disk
if modal.is_local() and not key:
    raise SystemExit("set DEMO_API_KEY first (see the docstring)")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_dict({"VLLM_API_KEY": key or ""})],
    timeout=3600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=900)
def serve() -> None:
    subprocess.Popen(["bash", "/root/serve/serve_vllm.sh", CHECKPOINT])
