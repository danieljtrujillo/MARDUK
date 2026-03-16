#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MARDUK Entrypoint — Persistent storage setup + web dashboard launch
# ---------------------------------------------------------------------------
set -euo pipefail

APP_DIR="/app/marduk"
PERSIST="/workspace"

echo "======================================"
echo " MARDUK — Container Startup"
echo "======================================"

# ── Symlink persistent directories to /workspace ──
# These survive pod restarts / stops:
#   /workspace/outputs   — checkpoints, metrics, predictions
#   /workspace/data      — processed data, augmented data
#   /workspace/logs      — training logs
for dir in outputs data/processed data/augmented; do
    persist_path="${PERSIST}/${dir}"
    app_path="${APP_DIR}/${dir}"
    mkdir -p "${persist_path}"
    # Remove the directory from the image (if exists) and symlink to persistent storage
    rm -rf "${app_path}"
    ln -sfn "${persist_path}" "${app_path}"
    echo "  ✓ ${dir} → ${persist_path}"
done

# Ensure raw data dir exists in the app (it's baked into the image, not persisted)
# But copy raw data to persistent storage on first run so it's not lost
if [ ! -f "${PERSIST}/data/raw/.initialized" ]; then
    echo "  Copying raw data to persistent storage (first run)..."
    mkdir -p "${PERSIST}/data/raw"
    cp -rn "${APP_DIR}/data/raw/." "${PERSIST}/data/raw/" 2>/dev/null || true
    touch "${PERSIST}/data/raw/.initialized"
    echo "  ✓ Raw data copied to ${PERSIST}/data/raw"
fi
# Symlink raw data too
rm -rf "${APP_DIR}/data/raw"
ln -sfn "${PERSIST}/data/raw" "${APP_DIR}/data/raw"

# ── Persist HuggingFace model cache across pod restarts ──
HF_CACHE_PERSIST="${PERSIST}/hf_cache"
mkdir -p "${HF_CACHE_PERSIST}"
export HF_HOME="${HF_CACHE_PERSIST}"
export TRANSFORMERS_CACHE="${HF_CACHE_PERSIST}"
export HF_TOKEN="${HF_TOKEN:-}"
echo "  ✓ HF_HOME → ${HF_CACHE_PERSIST}"

# ── Start SSH server (RunPod expects this for SSH access) ──
if [ -f /etc/init.d/ssh ]; then
    /etc/init.d/ssh start 2>/dev/null || service ssh start 2>/dev/null || true
    echo "  \u2713 SSH server started"
fi

echo ""
echo "  Launching MARDUK Web Dashboard on port 8888..."
echo "  Access via RunPod HTTP Service link."
echo "======================================"

cd "${APP_DIR}"
exec python web_dashboard.py --host 0.0.0.0 --port 8888
