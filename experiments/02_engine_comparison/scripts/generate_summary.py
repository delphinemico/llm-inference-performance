#!/usr/bin/env python3

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "summary.csv"

ENGINES = ["vllm", "sglang", "trtllm"]

WORKLOADS = {
    "interactive_c4": {
        "input_tokens": 512,
        "output_tokens": 128,
        "concurrency": 4,
    },
    "high_concurrency_c16": {
        "input_tokens": 512,
        "output_tokens": 128,
        "concurrency": 16,
    },
    "prefill_heavy": {
        "input_tokens": 4096,
        "output_tokens": 64,
        "concurrency": 4,
    },
    "decode_heavy": {
        "input_tokens": 512,
        "output_tokens": 512,
        "concurrency": 4,
    },
}


def metric(data, key, stat="avg"):
    value = data.get(key)
    if not isinstance(value, dict):
        return None
    return value.get(stat)


def rounded(value, digits=2):
    if value is None:
        return None
    return round(float(value), digits)


rows = []

for engine in ENGINES:
    for workload, config in WORKLOADS.items():
        path = RESULTS / engine / workload / "profile_export_aiperf.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")

        with path.open() as f:
            data = json.load(f)

        rows.append(
            {
                "engine": engine,
                "workload": workload,
                "input_tokens": config["input_tokens"],
                "output_tokens": config["output_tokens"],
                "concurrency": config["concurrency"],
                "aiperf_version": data.get("aiperf_version"),
                "completed_requests": rounded(
                    metric(data, "completed_request_count"), 0
                ),
                "error_rate_pct": rounded(
                    metric(data, "request_error_rate"), 2
                ),
                "request_throughput_rps": rounded(
                    metric(data, "request_throughput"), 2
                ),
                "output_throughput_tps": rounded(
                    metric(data, "output_token_throughput"), 2
                ),
                "input_throughput_tps": rounded(
                    metric(data, "input_token_throughput"), 2
                ),
                "median_ttft_ms": rounded(
                    metric(data, "time_to_first_token", "p50"), 2
                ),
                "p99_ttft_ms": rounded(
                    metric(data, "time_to_first_token", "p99"), 2
                ),
                "median_itl_ms": rounded(
                    metric(data, "inter_token_latency", "p50"), 2
                ),
                "median_request_latency_ms": rounded(
                    metric(data, "request_latency", "p50"), 2
                ),
                "avg_gpu_power_w": rounded(
                    metric(data, "nvidia_average_gpu_power"), 2
                ),
                "prompt_cache_read_pct": rounded(
                    metric(data, "overall_usage_prompt_cache_read_pct"), 2
                ),
            }
        )

fieldnames = list(rows[0].keys())

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
