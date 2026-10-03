import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

shared = ("You are an enterprise AI assistant. Follow policy, preserve factual accuracy, "
          "and reason only from the supplied context. ") * 220

suffixes = [
    "Summarize the key risk in one sentence.",
    "Identify the most important operational constraint.",
    "State the recommended next action.",
    "List the two most important facts.",
    "Explain the primary tradeoff briefly.",
]

def call(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "min_tokens": 32,
        "ignore_eos": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        body = json.loads(r.read().decode())
    elapsed = time.perf_counter() - t0
    usage = body.get("usage", {})
    details = usage.get("prompt_tokens_details", {}) or {}
    print({
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    })

for i, suffix in enumerate(suffixes, 1):
    print(f"request {i}")
    call(shared + "\n\nTask: " + suffix)
