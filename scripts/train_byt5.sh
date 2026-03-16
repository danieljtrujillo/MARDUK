#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Fine-tune ByT5-base for Akkadian→English
python -m src.train.train_byt5 \
  --data-config configs/data/raw.yaml \
  --view-config configs/data/dual_view.yaml \
  --model-config configs/model/byt5_base.yaml \
  --train-config configs/train/byt5_finetune.yaml
