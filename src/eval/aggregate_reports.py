from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    reports = []
    for metrics_path in runs_dir.glob("*/metrics.json"):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        reports.append({"run": metrics_path.parent.name, **metrics})
    reports = sorted(reports, key=lambda x: x.get("sacrebleu", -1), reverse=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
