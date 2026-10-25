"""OpenAI-compatible vLLM endpoint on Modal for the net-influence experiments (written during the sprint).

Deploy once:   modal deploy serve_qwen.py
The app scales to zero after 5 idle minutes, so it only costs credits while games are running.
Clients authenticate with the key stored in the Modal secret `ni-endpoint` (MODAL_SERVE_API_KEY).
"""
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"  # open weights, no Hugging Face gate
GPU = "A10G"
PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("ni-qwen-serve")


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * MINUTES,
    scaledown_window=5 * MINUTES,
    secrets=[modal.Secret.from_name("ni-endpoint")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=15 * MINUTES)
def serve():
    import os

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--served-model-name", MODEL,
        "--max-model-len", "8192",
        "--enforce-eager",  # skip CUDA-graph capture: faster cold starts, fine for short game turns
        "--api-key", os.environ["MODAL_SERVE_API_KEY"],
    ]
    subprocess.Popen(cmd)
