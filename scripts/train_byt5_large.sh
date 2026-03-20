#!/usr/bin/env bash
set -euo pipefail

# Phase 2: Fine-tune ByT5-large for Akkadian→English on B200
# Step 1: Prepare data (applies cleaning + truncation recovery)
echo "=== Preparing data ==="
python -m src.data.prepare \
  --data-config configs/data/raw.yaml \
  --view-config configs/data/dual_view.yaml

# Step 2: Train byt5-large
echo "=== Training ByT5-large ==="
python -m src.train.train_byt5 \
  --data-config configs/data/raw.yaml \
  --view-config configs/data/dual_view.yaml \
  --model-config configs/model/byt5_large.yaml \
  --train-config configs/train/byt5_large_b200.yaml

echo "=== Training complete ==="
echo "Best model saved to: outputs/runs/byt5_large_v1/best/"
