#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate a consolidated CSV from vLLM benchmark JSON results."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark JSON files.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    study_dir = script_dir.parent

    result_dir = (
        args.result_dir
        if args.result_dir is not None
        else study_dir / "results" / "L40S_2026-09-26"
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for filename in sorted(glob.glob(str(result_dir / "*.json"))):
        path = Path(filename)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row = {"experiment": path.stem}

        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                key_lower = key.lower()

                if any(
                    metric in key_lower
                    for metric in (
                        "throughput",
                        "ttft",
                        "tpot",
                        "itl",
                        "e2el",
                        "latency",
                        "successful",
                        "failed",
                    )
                ):
                    row[key] = value

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No benchmark JSON files found in {result_dir}")

    df = pd.DataFrame(rows)

    output_path = result_dir / "summary.csv"
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()