"""MARDUK Web Dashboard — FastAPI + WebSocket backend for RunPod deployment.

Launch:
    python web_dashboard.py [--port 8888] [--host 0.0.0.0]
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

app = FastAPI(title="MARDUK Dashboard")

# Allow cross-origin requests from the standalone dashboard.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Process manager — one process at a time, streams output to WebSocket clients
# ---------------------------------------------------------------------------
# Regex to extract eval metrics from HF Trainer log lines
_METRIC_RE = re.compile(r"'(eval_[a-z_]+|loss|learning_rate|epoch|grad_norm)'\s*:\s*([\d.eE+-]+)")
# Regex to parse tqdm-style progress bars:  5%|█| 50/1000 [02:30<47:30, 3.00s/it]
_TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+)")


class ProcessManager:
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.clients: list[WebSocket] = []
        self.log_buffer: list[str] = []
        self.running = False
        self.current_task = ""
        self.start_time: float = 0.0         # process start timestamp
        self.progress: dict = {}             # latest progress info
        self.live_train_metrics: dict = {}   # latest training step metrics
        self.live_eval_metrics: dict = {}    # latest eval metrics
        self.best_live_score: float = 0.0    # best competition_score seen live

    @staticmethod
    def _parse_time_str(s: str) -> int:
        """Parse HH:MM:SS or MM:SS to total seconds."""
        parts = s.split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]

    def _parse_progress_from_log(self, text: str):
        """Extract tqdm progress info from log lines."""
        m = _TQDM_RE.search(text)
        if not m:
            return
        pct, step, total, elapsed_s, remaining_s = m.groups()
        elapsed_sec = self._parse_time_str(elapsed_s)
        remaining_sec = self._parse_time_str(remaining_s)
        self.progress = {
            "pct": int(pct),
            "step": int(step),
            "total": int(total),
            "elapsed": elapsed_s,
            "elapsed_sec": elapsed_sec,
            "eta": remaining_s,
            "eta_sec": remaining_sec,
        }

    def _parse_metrics_from_log(self, text: str):
        """Extract metrics from HF Trainer log lines like {'loss': 2.3, ...}."""
        pairs = _METRIC_RE.findall(text)
        if not pairs:
            return
        parsed = {}
        for key, val in pairs:
            try:
                parsed[key] = float(val)
            except ValueError:
                pass
        if not parsed:
            return
        has_eval = any(k.startswith("eval_") for k in parsed)
        if has_eval:
            self.live_eval_metrics.update(parsed)
            score = parsed.get("eval_competition_score", 0.0)
            if score > self.best_live_score:
                self.best_live_score = score
        else:
            self.live_train_metrics.update(parsed)

    async def broadcast(self, message: dict):
        self.log_buffer.append(json.dumps(message))
        # Parse metrics and progress from log lines
        if message.get("type") == "log":
            text = message.get("text", "")
            self._parse_metrics_from_log(text)
            self._parse_progress_from_log(text)
        # Keep last 5000 lines
        if len(self.log_buffer) > 5000:
            self.log_buffer = self.log_buffer[-5000:]
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)

    async def run_command(self, cmd: list[str], task_name: str = ""):
        if self.running:
            await self.broadcast({"type": "error", "text": "A process is already running."})
            return
        self.running = True
        self.current_task = task_name
        self.start_time = time.time()
        self.log_buffer.clear()
        self.progress.clear()
        self.live_train_metrics.clear()
        self.live_eval_metrics.clear()
        self.best_live_score = 0.0
        await self.broadcast({"type": "started", "task": task_name})

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(ROOT),
            )
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                await self.broadcast({"type": "log", "text": text})

            await self.process.wait()
            code = self.process.returncode
            await self.broadcast({"type": "finished", "code": code, "task": task_name})
        except Exception as e:
            await self.broadcast({"type": "error", "text": str(e)})
        finally:
            self.running = False
            self.process = None

    async def kill(self):
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(2)
                if self.process and self.process.returncode is None:
                    self.process.kill()
            except ProcessLookupError:
                pass
            await self.broadcast({"type": "killed", "task": self.current_task})
            self.running = False

pm = ProcessManager()


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def detect_hardware() -> dict:
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpus": [],
            "bf16": False,
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "mem_gb": round(props.total_memory / 1024**3, 1),
                })
            info["bf16"] = torch.cuda.is_bf16_supported()
        return info
    except Exception:
        return {
            "torch_version": "?",
            "cuda_available": False,
            "cuda_version": None,
            "gpus": [],
            "bf16": False,
        }


_hw_info = None

def get_hw():
    global _hw_info
    if _hw_info is None:
        _hw_info = detect_hardware()
    return _hw_info


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/hardware")
async def api_hardware():
    return get_hw()


@app.get("/api/status")
async def api_status():
    result = {
        "running": pm.running,
        "task": pm.current_task,
        "start_time": pm.start_time,
        "elapsed_sec": round(time.time() - pm.start_time) if pm.running else 0,
        "progress": pm.progress,
    }
    return result


@app.get("/api/metrics")
async def api_metrics():
    """Return best metrics across all runs + live training metrics."""
    runs = {}
    best = {}
    best_score = -1.0
    for p in sorted(ROOT.glob("outputs/runs/*/metrics.json")):
        try:
            m = json.loads(p.read_text())
            run_name = p.parent.name
            runs[run_name] = m
            score = m.get("eval_competition_score", m.get("competition_score", 0.0))
            if score > best_score:
                best_score = score
                best = m.copy()
                best["_run"] = run_name
        except Exception:
            continue
    # Normalize key names (HF Trainer prefix eval_ keys)
    for key in list(best.keys()):
        if key.startswith("eval_"):
            short = key[5:]
            if short not in best:
                best[short] = best[key]
    result = {
        "best": best,
        "best_score": best_score,
        "runs": runs,
        "live": {
            "training": pm.live_train_metrics.copy(),
            "eval": pm.live_eval_metrics.copy(),
            "best_live_score": pm.best_live_score,
            "is_training": pm.running,
            "current_task": pm.current_task,
            "progress": pm.progress.copy(),
            "elapsed_sec": round(time.time() - pm.start_time) if pm.running else 0,
        },
    }
    return result


@app.get("/api/checkpoints")
async def api_checkpoints():
    """List available checkpoint files."""
    pts = sorted(ROOT.glob("outputs/runs/*/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": str(p.relative_to(ROOT)), "size_mb": round(p.stat().st_size / 1024**2, 1)} for p in pts[:20]]


class RunRequest(BaseModel):
    task: str
    args: dict = {}


@app.post("/api/run")
async def api_run(req: RunRequest):
    if pm.running:
        return JSONResponse({"error": "A process is already running"}, status_code=409)

    cmd = _build_command(req.task, req.args)
    if not cmd:
        return JSONResponse({"error": f"Unknown task: {req.task}"}, status_code=400)

    asyncio.create_task(pm.run_command(cmd, req.task))
    return {"status": "started", "task": req.task}


@app.post("/api/kill")
async def api_kill():
    await pm.kill()
    return {"status": "killed"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pm.clients.append(ws)
    # Send buffered logs
    for msg in pm.log_buffer:
        try:
            await ws.send_text(msg)
        except Exception:
            break
    # Send current status
    await ws.send_json({"type": "status", "running": pm.running, "task": pm.current_task})
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        if ws in pm.clients:
            pm.clients.remove(ws)


def _build_command(task: str, args: dict) -> list[str] | None:
    if task == "prepare":
        return [
            PYTHON, "-m", "src.data.prepare",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
        ]
    elif task == "train":
        cmd = [
            PYTHON, "-m", "src.train.train_hybrid",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
            "--model-config", args.get("model_config", "configs/model/mamba_enc_txd_dec_base.yaml"),
            "--train-config", args.get("train_config", "configs/train/b200.yaml"),
            "--device", args.get("device", "cuda"),
        ]
        if args.get("bf16", True):
            cmd.append("--bf16")
        return cmd
    elif task == "train_byt5":
        return [
            PYTHON, "-m", "src.train.train_byt5",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
            "--model-config", args.get("model_config", "configs/model/byt5_base.yaml"),
            "--train-config", args.get("train_config", "configs/train/byt5_finetune.yaml"),
        ]
    elif task == "train_byt5_expanded":
        return [
            PYTHON, "-m", "src.train.train_byt5",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
            "--model-config", args.get("model_config", "configs/model/byt5_base.yaml"),
            "--train-config", args.get("train_config", "configs/train/byt5_expanded.yaml"),
        ]
    elif task == "expand_data":
        return [
            PYTHON, "-m", "src.data.expand_training",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
        ]
    elif task == "train_mamba_byt5":
        return [
            PYTHON, "-m", "src.train.train_mamba_byt5",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
            "--model-config", args.get("model_config", "configs/model/mamba_byt5.yaml"),
            "--train-config", args.get("train_config", "configs/train/mamba_byt5_finetune.yaml"),
        ]
    elif task == "inference":
        return [
            PYTHON, "-m", "src.eval.decode",
            "--data-config", args.get("data_config", "configs/data/raw.yaml"),
            "--view-config", args.get("view_config", "configs/data/dual_view.yaml"),
            "--model-config", args.get("model_config", "configs/model/mamba_enc_txd_dec_base.yaml"),
            "--checkpoint", args.get("checkpoint", "outputs/runs/hybrid_b200/best.pt"),
            "--output", args.get("output", "submission.csv"),
            "--batch-size", str(args.get("batch_size", 8)),
            "--device", args.get("device", "cuda"),
            "--num-beams", str(args.get("num_beams", 5)),
        ]
    elif task == "evaluate":
        return [
            PYTHON, "-m", "src.eval.aggregate_reports",
            "--runs-dir", args.get("runs_dir", "outputs/runs"),
            "--out", args.get("output", "outputs/reports/summary.json"),
        ]
    elif task == "pipeline":
        return ["bash", str(ROOT / "runpod_start.sh")]
    return None


# ---------------------------------------------------------------------------
# File browsing & download (for pulling reports from RunPod)
# ---------------------------------------------------------------------------
ALLOWED_DIRS = ["outputs", "data/processed", "data/augmented", "submission.csv"]
WORKSPACE = Path("/workspace")


def _validate_path(rel_path: str) -> Path | None:
    """Validate and resolve a relative path against allowed directories.
    Returns the resolved absolute path or None if invalid."""
    if not any(rel_path.startswith(a) for a in ALLOWED_DIRS):
        return None
    # Build path from ROOT (may follow symlinks to /workspace)
    target = ROOT / rel_path
    if not target.exists():
        return None
    resolved = target.resolve()
    # Allow paths under ROOT or /workspace (where symlinks point)
    if str(resolved).startswith(str(ROOT.resolve())) or str(resolved).startswith(str(WORKSPACE.resolve())):
        return resolved
    return None


@app.get("/api/files/list")
async def api_files_list(dir: str = "outputs"):
    """List files under an allowed directory."""
    target = ROOT / dir
    if not any(dir.startswith(a) for a in ALLOWED_DIRS):
        return JSONResponse({"error": "Directory not allowed"}, status_code=403)
    if not target.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    if target.is_file():
        return [{"path": dir, "size_mb": round(target.stat().st_size / 1024**2, 2), "type": "file"}]
    items = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            items.append({"path": rel, "size_mb": round(p.stat().st_size / 1024**2, 2)})
    return items


@app.get("/api/files/download")
async def api_files_download(path: str):
    """Download a file from an allowed directory."""
    if not any(path.startswith(a) for a in ALLOWED_DIRS):
        return JSONResponse({"error": "Path not allowed"}, status_code=403)
    target = ROOT / path
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(target.resolve()), filename=target.name)


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


# Mount static files
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    cli_args = parser.parse_args()
    uvicorn.run(app, host=cli_args.host, port=cli_args.port, log_level="info")
