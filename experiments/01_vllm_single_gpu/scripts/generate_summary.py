#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_RESULT_DIR = BENCHMARK_DIR / "results" / "local"


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def flatten_scalars(obj, prefix=""):
    scalars = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                scalars.update(flatten_scalars(value, full_key))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                scalars[full_key] = value

    return scalars


def should_include(key: str) -> bool:
    key_lower = key.lower()

    metric_terms = (
        "throughput",
        "ttft",
        "tpot",
        "itl",
        "e2el",
        "latency",
        "successful",
        "failed",
        "completed",
    )

    metadata_terms = (
        "num_prompts",
        "max_concurrency",
        "input_len",
        "output_len",
        "model",
        "seed",
        "request_rate",
    )

    return any(term in key_lower for term in metric_terms + metadata_terms)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate summary.csv from vLLM benchmark JSON results."
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Directory containing benchmark JSON files. Default: {DEFAULT_RESULT_DIR}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <result-dir>/summary.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = args.result_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else result_dir / "summary.csv"
    )

    if not result_dir.exists():
        raise SystemExit(
            f"Result directory does not exist: {result_dir}\n"
            "Run the benchmark scripts first or provide --result-dir."
        )

    json_files = sorted(result_dir.glob("*.json"), key=natural_sort_key)

    if not json_files:
        raise SystemExit(
            f"No JSON benchmark files found in: {result_dir}"
        )

    rows = []
    all_columns = set()

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        flattened = flatten_scalars(data)

        row = {"experiment": json_path.stem}

        for key, value in flattened.items():
            if should_include(key):
                row[key] = value

        rows.append(row)
        all_columns.update(row.keys())

    preferred_columns = [
        "experiment",
        "model_id",
        "model",
        "num_prompts",
        "max_concurrency",
        "request_rate",
        "completed",
        "successful",
        "failed",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p99_e2el_ms",
    ]

    columns = []

    for column in preferred_columns:
        if column in all_columns:
            columns.append(column)

    remaining_columns = sorted(all_columns - set(columns))
    columns.extend(remaining_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Read {len(json_files)} benchmark JSON files")
    print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()