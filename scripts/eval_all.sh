#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.aggregate_reports --runs-dir outputs/runs --out outputs/reports/summary.json
