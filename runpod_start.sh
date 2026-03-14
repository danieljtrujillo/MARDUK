#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RunPod startup script — runs the full MARDUK pipeline
# Usage:  bash runpod_start.sh [--skip-prep] [--epochs 15] [--batch-size 32]
# ---------------------------------------------------------------------------
set -euo pipefail

cd /app/marduk

SKIP_PREP=false
EPOCHS=""
BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-prep)   SKIP_PREP=true; shift ;;
    --epochs)      EPOCHS="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "======================================"
echo " MARDUK — RunPod Pipeline"
echo "======================================"
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}  |  {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB')"
echo "--------------------------------------"

# 1. Data preparation
if [ "$SKIP_PREP" = false ]; then
  echo "[1/3] Preparing data..."
  python -m src.data.prepare \
    --data-config configs/data/raw.yaml \
    --view-config configs/data/dual_view.yaml
else
  echo "[1/3] Skipping data prep (--skip-prep)"
fi

# 2. Training
echo "[2/3] Training hybrid model..."
EXTRA_ARGS=""
[ -n "$EPOCHS" ]     && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[ -n "$BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"

python -m src.train.train_hybrid \
  --data-config  configs/data/raw.yaml \
  --view-config  configs/data/dual_view.yaml \
  --model-config configs/model/mamba_enc_txd_dec_base.yaml \
  --train-config configs/train/b200.yaml \
  --device cuda --bf16 \
  $EXTRA_ARGS

# 3. Evaluation
echo "[3/3] Aggregating reports..."
python -m src.eval.aggregate_reports \
  --runs-dir outputs/runs \
  --out outputs/reports/summary.json

echo "======================================"
echo " Pipeline complete!"
echo "======================================"
