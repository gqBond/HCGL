"""Aggregate JSON results produced by main.py."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in sorted(args.input_dir.glob("*_seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        grouped[(result["dataset"], result["method"])].append(result)

    if not grouped:
        raise SystemExit(f"No result JSON files found in {args.input_dir}")

    rows = []
    for (dataset, method), results in sorted(grouped.items()):
        performance = [item["average_performance"] for item in results]
        forgetting = [item["average_forgetting"] for item in results]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(results),
                "seeds": " ".join(str(item["seed"]) for item in results),
                "ap_mean": statistics.mean(performance),
                "ap_std": statistics.stdev(performance) if len(performance) > 1 else 0.0,
                "af_mean": statistics.mean(forgetting),
                "af_std": statistics.stdev(forgetting) if len(forgetting) > 1 else 0.0,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
