#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_hybrid \
  --data-config configs/data/raw.yaml \
  --view-config configs/data/dual_view.yaml \
  --model-config configs/model/mamba_enc_txd_dec_base.yaml \
  --train-config configs/train/hybrid.yaml
