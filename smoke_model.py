"""Smoke test: one chat completion through poseidon_compute against the configured endpoint.

Waits through a Modal cold start (model download + vLLM boot) before sending the prompt.
"""
import time

from poseidon_compute import make_client, resolve_model

client = make_client()
model = resolve_model("Qwen/Qwen2.5-7B-Instruct")

deadline = time.time() + 20 * 60
while True:
    try:
        client.models.list()
        break
    except Exception as e:  # cold start: connection errors / 503s until vLLM is up
        if time.time() > deadline:
            raise
        print("waiting for endpoint:", type(e).__name__, flush=True)
        time.sleep(15)

start = time.time()
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Pick rock, paper or scissors. Answer with one word."}],
    max_tokens=5,
    temperature=0,
)
print("MODEL:", model)
print("REPLY:", resp.choices[0].message.content)
print(f"LATENCY: {time.time() - start:.2f}s")
