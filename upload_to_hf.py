#!/usr/bin/env python3
"""Upload model weights from RunPod to HuggingFace Hub.

Usage (from local machine):
    python upload_to_hf.py

What it does:
    1. Connects to RunPod via the dashboard API
    2. Writes a small upload script to the pod
    3. Runs it (installs huggingface_hub if needed, uploads folder)
    4. Polls the log until UPLOAD COMPLETE or UPLOAD FAILED
    5. Prints the result

Config is read from environment or prompted interactively.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

# ── defaults (override via env vars) ──
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "danieljtrujillo/marduk-byt5-akkadian2english")
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/outputs/runs/byt5_plain_v1/best")
DASHBOARD_PORT = os.environ.get("DASHBOARD_PORT", "8888")


def _api_base(pod_id: str, port: str) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net/api"


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _post_json(url: str, data: dict, headers: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get_text(url: str, headers: dict, timeout: int = 30) -> str:
    h = {k: v for k, v in headers.items() if k != "Content-Type"}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _prompt(name: str, default: str, secret: bool = False) -> str:
    if default:
        return default
    val = input(f"{name}: ").strip()
    if not val:
        print(f"ERROR: {name} is required", file=sys.stderr)
        sys.exit(1)
    return val


def run_upload(
    pod_id: str,
    api_key: str,
    hf_token: str,
    hf_repo: str,
    model_dir: str,
    port: str = "8888",
) -> bool:
    """Run the upload on RunPod. Returns True on success."""
    base = _api_base(pod_id, port)
    hdrs = _headers(api_key)

    # ── Step 1: Write upload script to pod ──
    upload_script = f'''
import os, sys
try:
    from huggingface_hub import HfApi
except ImportError:
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import HfApi

token = "{hf_token}"
repo  = "{hf_repo}"
folder = "{model_dir}"

print(f"Uploading {{folder}} -> {{repo}}")
if not os.path.isdir(folder):
    print(f"UPLOAD FAILED: {{folder}} does not exist")
    sys.exit(1)

files = os.listdir(folder)
total = sum(os.path.getsize(os.path.join(folder, f)) for f in files)
print(f"  {{len(files)}} files, {{total / 1e9:.2f}} GB")

api = HfApi()
api.create_repo(repo_id=repo, token=token, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path=folder,
    repo_id=repo,
    token=token,
    repo_type="model",
    commit_message="Model update from RunPod",
)
print("UPLOAD COMPLETE")
'''

    log_path = "/workspace/outputs/hf_upload.log"

    # Write script
    write_cmd = (
        f"cat > /workspace/_hf_upload.py << 'UPLOADEOF'\n"
        f"{upload_script}\n"
        f"UPLOADEOF\n"
        f"echo 'Script written'"
    )
    print("[1/4] Writing upload script to pod...")
    _post_json(f"{base}/run", {"task": "shell", "args": {"cmd": write_cmd}}, hdrs)

    # Wait for write to finish
    for _ in range(10):
        time.sleep(1)
        status = _post_json(f"{base}/status", {}, hdrs)
        if not status.get("running", True):
            break

    # ── Step 2: Launch upload (background via nohup) ──
    run_cmd = (
        f"nohup python3 /workspace/_hf_upload.py "
        f"> {log_path} 2>&1 &"
    )
    print("[2/4] Launching upload...")
    _post_json(f"{base}/run", {"task": "shell", "args": {"cmd": run_cmd}}, hdrs)
    time.sleep(2)

    # ── Step 3: Poll log until done ──
    print("[3/4] Monitoring upload progress...")
    log_url = f"{base}/files/download?path={urllib.parse.quote(log_path.lstrip('/workspace/'))}"

    max_wait = 600  # 10 minutes max
    poll_interval = 10
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            log_text = _get_text(log_url, hdrs, timeout=15)
        except Exception:
            print(f"  [{elapsed}s] Waiting for log file...")
            continue

        # Show last meaningful line
        lines = [l for l in log_text.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1][:120]
            print(f"  [{elapsed}s] {last}")

        if "UPLOAD COMPLETE" in log_text:
            print("\n[4/4] Upload successful!")
            return True
        if "UPLOAD FAILED" in log_text:
            print(f"\n[4/4] Upload FAILED!")
            print(log_text)
            return False

    print(f"\nTimeout after {max_wait}s. Check log manually: {log_path}")
    return False


def main():
    pod_id = _prompt("RUNPOD_POD_ID", RUNPOD_POD_ID)
    api_key = _prompt("RUNPOD_API_KEY", RUNPOD_API_KEY)
    hf_token = _prompt("HF_TOKEN", HF_TOKEN)
    hf_repo = _prompt("HF_REPO", HF_REPO)
    model_dir = _prompt("MODEL_DIR", MODEL_DIR)
    port = _prompt("DASHBOARD_PORT", DASHBOARD_PORT)

    print(f"\n{'='*60}")
    print(f"Pod:       {pod_id}")
    print(f"HF Repo:   {hf_repo}")
    print(f"Model Dir: {model_dir}")
    print(f"{'='*60}\n")

    ok = run_upload(pod_id, api_key, hf_token, hf_repo, model_dir, port)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
