#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent

DEFAULT_RESULT_DIR = BENCHMARK_DIR / "results" / "local"
DEFAULT_FIGURE_DIR = BENCHMARK_DIR / "figures" / "local"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate figures from vLLM benchmark JSON results."
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Directory containing benchmark JSON files. Default: {DEFAULT_RESULT_DIR}",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_FIGURE_DIR}",
    )

    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark result: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(data: dict, *candidate_keys):
    for key in candidate_keys:
        if key in data:
            return data[key]

    raise KeyError(
        "Could not find any of these metric keys: "
        + ", ".join(candidate_keys)
    )


def save_figure(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_concurrency_throughput(result_dir: Path, figure_dir: Path):
    concurrencies = [1, 4, 8, 16]
    throughput = []

    for concurrency in concurrencies:
        data = load_json(
            result_dir / f"e1_concurrency_{concurrency}.json"
        )

        throughput.append(
            get_metric(
                data,
                "output_throughput",
                "output_token_throughput",
            )
        )

    plt.figure()
    plt.plot(concurrencies, throughput, marker="o")
    plt.xticks(concurrencies)
    plt.xlabel("Max concurrency")
    plt.ylabel("Output throughput (tokens/s)")
    plt.title("Output Throughput vs Concurrency")
    plt.grid(True, alpha=0.3)

    save_figure(
        figure_dir / "concurrency_throughput.png"
    )


def plot_concurrency_ttft(result_dir: Path, figure_dir: Path):
    concurrencies = [1, 4, 8, 16]
    median_ttft = []

    for concurrency in concurrencies:
        data = load_json(
            result_dir / f"e1_concurrency_{concurrency}.json"
        )

        median_ttft.append(
            get_metric(
                data,
                "median_ttft_ms",
                "median_ttft",
            )
        )

    plt.figure()
    plt.plot(concurrencies, median_ttft, marker="o")
    plt.xticks(concurrencies)
    plt.xlabel("Max concurrency")
    plt.ylabel("Median TTFT (ms)")
    plt.title("Time to First Token vs Concurrency")
    plt.grid(True, alpha=0.3)

    save_figure(
        figure_dir / "concurrency_ttft.png"
    )


def plot_prefill_ttft(result_dir: Path, figure_dir: Path):
    files = [
        ("512", result_dir / "e2_prefill_512x64.json"),
        ("4096", result_dir / "e2_prefill_4096x64.json"),
    ]

    labels = []
    median_ttft = []

    for label, path in files:
        data = load_json(path)

        labels.append(label)
        median_ttft.append(
            get_metric(
                data,
                "median_ttft_ms",
                "median_ttft",
            )
        )

    plt.figure()
    plt.bar(labels, median_ttft)
    plt.xlabel("Input tokens")
    plt.ylabel("Median TTFT (ms)")
    plt.title("Prefill Scaling")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(
        figure_dir / "prefill_ttft.png"
    )


def plot_decode_e2e_latency(result_dir: Path, figure_dir: Path):
    files = [
        ("64", result_dir / "e3_decode_512x64.json"),
        ("512", result_dir / "e3_decode_512x512.json"),
    ]

    labels = []
    median_e2e = []

    for label, path in files:
        data = load_json(path)

        labels.append(label)
        median_e2e.append(
            get_metric(
                data,
                "median_e2el_ms",
                "median_e2e_latency_ms",
                "median_e2e_latency",
            )
        )

    plt.figure()
    plt.bar(labels, median_e2e)
    plt.xlabel("Output tokens")
    plt.ylabel("Median end-to-end latency (ms)")
    plt.title("Decode Scaling")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(
        figure_dir / "decode_e2e_latency.png"
    )


def plot_prefix_cache_ttft(result_dir: Path, figure_dir: Path):
    off_path = result_dir / "e4_prefix_cache_OFF.json"
    on_path = result_dir / "e4_prefix_cache_ON.json"

    if not off_path.exists():
        off_path = result_dir / "e4_prefix_cache_off.json"

    if not on_path.exists():
        on_path = result_dir / "e4_prefix_cache_on.json"

    off_data = load_json(off_path)
    on_data = load_json(on_path)

    median_ttft = [
        get_metric(
            off_data,
            "median_ttft_ms",
            "median_ttft",
        ),
        get_metric(
            on_data,
            "median_ttft_ms",
            "median_ttft",
        ),
    ]

    plt.figure()
    plt.bar(["OFF", "ON"], median_ttft)
    plt.xlabel("Automatic prefix caching")
    plt.ylabel("Median TTFT (ms)")
    plt.title("Prefix Caching")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(
        figure_dir / "prefix_cache_ttft.png"
    )


def main():
    args = parse_args()

    result_dir = args.result_dir.resolve()
    figure_dir = args.figure_dir.resolve()

    if not result_dir.exists():
        raise SystemExit(
            f"Result directory does not exist: {result_dir}\n"
            "Run the benchmark scripts first or provide --result-dir."
        )

    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_concurrency_throughput(result_dir, figure_dir)
    plot_concurrency_ttft(result_dir, figure_dir)
    plot_prefill_ttft(result_dir, figure_dir)
    plot_decode_e2e_latency(result_dir, figure_dir)
    plot_prefix_cache_ttft(result_dir, figure_dir)

    print(f"Figures written to: {figure_dir}")


if __name__ == "__main__":
    main()