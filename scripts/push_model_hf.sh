#!/usr/bin/env bash
set -euo pipefail

# Push trained ByT5-large model to HuggingFace
# Run on RunPod after training completes

MODEL_DIR="outputs/runs/byt5_large_v1/best"
REPO_ID="danieljtrujillo/marduk-byt5-akkadian2english"
HF_TOKEN="${HF_TOKEN:-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}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory $MODEL_DIR not found"
    exit 1
fi

echo "=== Pushing model to HuggingFace ==="
echo "  Model:  $MODEL_DIR"
echo "  Repo:   $REPO_ID"

python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi(token='${HF_TOKEN}')
api.upload_folder(
    folder_path='${MODEL_DIR}',
    repo_id='${REPO_ID}',
    repo_type='model',
    commit_message='byt5-large v1: retrained with cleaned data, label smoothing, 40 epochs',
)
print('Upload complete!')
"

echo "=== Done ==="
