#!/usr/bin/env bash
set -euo pipefail

python -m src.data.prepare \
  --data-config configs/data/raw.yaml \
  --view-config configs/data/dual_view.yaml
