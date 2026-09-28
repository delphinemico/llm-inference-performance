#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
RESULT_DIR = STUDY_DIR / "results" / "L40S_2026-09-26"
FIGURE_DIR = STUDY_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_result(filename):
    path = RESULT_DIR / filename

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(filename):
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Figure 1: Concurrency vs output-token throughput
# ---------------------------------------------------------

concurrency_runs = {
    1: load_result("e1_concurrency_1.json"),
    4: load_result("e1_concurrency_4.json"),
    8: load_result("e1_concurrency_8.json"),
    16: load_result("e1_concurrency_16.json"),
}

concurrency = list(concurrency_runs.keys())

output_throughput = [
    concurrency_runs[c]["output_throughput"]
    for c in concurrency
]

plt.figure(figsize=(7, 4.5))
plt.plot(concurrency, output_throughput, marker="o")
plt.xlabel("Maximum concurrency")
plt.ylabel("Output throughput (tokens/s)")
plt.title("Output Throughput vs Concurrency")
plt.xticks(concurrency)
plt.grid(axis="y", alpha=0.25)

save_figure("concurrency_throughput.png")


# ---------------------------------------------------------
# Figure 2: Concurrency vs TTFT
# ---------------------------------------------------------

median_ttft = [
    concurrency_runs[c]["median_ttft_ms"]
    for c in concurrency
]

p99_ttft = [
    concurrency_runs[c]["p99_ttft_ms"]
    for c in concurrency
]

plt.figure(figsize=(7, 4.5))
plt.plot(concurrency, median_ttft, marker="o", label="Median TTFT")
plt.plot(concurrency, p99_ttft, marker="o", label="P99 TTFT")
plt.xlabel("Maximum concurrency")
plt.ylabel("TTFT (ms)")
plt.title("Time to First Token vs Concurrency")
plt.xticks(concurrency)
plt.legend()
plt.grid(axis="y", alpha=0.25)

save_figure("concurrency_ttft.png")


# ---------------------------------------------------------
# Figure 3: Input length vs TTFT
# ---------------------------------------------------------

prefill_short = load_result("e2_prefill_512x64.json")
prefill_long = load_result("e2_prefill_4096x64.json")

input_lengths = ["512", "4096"]

prefill_median_ttft = [
    prefill_short["median_ttft_ms"],
    prefill_long["median_ttft_ms"],
]

plt.figure(figsize=(6, 4.5))
bars = plt.bar(input_lengths, prefill_median_ttft)

plt.xlabel("Input tokens")
plt.ylabel("Median TTFT (ms)")
plt.title("Prefill Scaling: Median TTFT")

for bar, value in zip(bars, prefill_median_ttft):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f} ms",
        ha="center",
        va="bottom",
    )

plt.grid(axis="y", alpha=0.25)

save_figure("prefill_ttft.png")


# ---------------------------------------------------------
# Figure 4: Output length vs end-to-end latency
# ---------------------------------------------------------

decode_short = load_result("e3_decode_512x64.json")
decode_long = load_result("e3_decode_512x512.json")

output_lengths = ["64", "512"]

decode_median_e2e_seconds = [
    decode_short["median_e2el_ms"] / 1000,
    decode_long["median_e2el_ms"] / 1000,
]

plt.figure(figsize=(6, 4.5))
bars = plt.bar(output_lengths, decode_median_e2e_seconds)

plt.xlabel("Output tokens")
plt.ylabel("Median end-to-end latency (s)")
plt.title("Decode Scaling: End-to-End Latency")

for bar, value in zip(bars, decode_median_e2e_seconds):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f} s",
        ha="center",
        va="bottom",
    )

plt.grid(axis="y", alpha=0.25)

save_figure("decode_e2e_latency.png")


# ---------------------------------------------------------
# Figure 5: Prefix caching vs TTFT
# ---------------------------------------------------------

prefix_off = load_result("e4_prefix_cache_OFF.json")
prefix_on = load_result("e4_prefix_cache_ON.json")

cache_modes = ["OFF", "ON"]

prefix_median_ttft = [
    prefix_off["median_ttft_ms"],
    prefix_on["median_ttft_ms"],
]

plt.figure(figsize=(6, 4.5))
bars = plt.bar(cache_modes, prefix_median_ttft)

plt.xlabel("Automatic prefix caching")
plt.ylabel("Median TTFT (ms)")
plt.title("Prefix Caching: Median TTFT")

for bar, value in zip(bars, prefix_median_ttft):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f} ms",
        ha="center",
        va="bottom",
    )

plt.grid(axis="y", alpha=0.25)

save_figure("prefix_cache_ttft.png")