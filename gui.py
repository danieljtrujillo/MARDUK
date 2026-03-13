"""MARDUK Control Panel — PyQt5 GUI for the full Akkadian→English MT pipeline.

Launch:
    python gui.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import queue
import signal
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# PyQt5 imports
# ---------------------------------------------------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QPushButton, QFileDialog, QTextEdit, QGroupBox,
    QSplitter, QProgressBar, QMessageBox, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor, QIcon, QPalette

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Default config paths (relative to ROOT)
DEFAULT_DATA_CFG = "configs/data/raw.yaml"
DEFAULT_VIEW_CFG = "configs/data/dual_view.yaml"
DEFAULT_MODEL_CFG = "configs/model/mamba_enc_txd_dec_base.yaml"
DEFAULT_TRAIN_CFG = "configs/train/hybrid.yaml"


# ---------------------------------------------------------------------------
# Hardware detection — run via subprocess to avoid DLL conflicts with PyQt5
# ---------------------------------------------------------------------------
def _detect_hw_subprocess() -> dict:
    """Detect hardware by running a small Python script in a subprocess.

    This avoids loading torch DLLs in the same process as PyQt5, which can
    cause DLL init failures on Windows.
    """
    script = (
        "import json, sys; "
        "info = {'torch_version':'?','cuda_available':False,'cuda_version':None,"
        "'gpus':[],'bf16':False,'default_device':'cpu','devices':['cpu']}; "
        "try:\n"
        "    import torch\n"
        "    info['torch_version'] = torch.__version__\n"
        "    info['cuda_available'] = torch.cuda.is_available()\n"
        "    info['cuda_version'] = getattr(torch.version, 'cuda', None)\n"
        "    devs = []\n"
        "    if torch.cuda.is_available():\n"
        "        for i in range(torch.cuda.device_count()):\n"
        "            props = torch.cuda.get_device_properties(i)\n"
        "            info['gpus'].append({'index':i,'name':props.name,"
        "'mem_gb':round(props.total_memory/1024**3,1)})\n"
        "            devs.append(f'cuda:{i}')\n"
        "        info['bf16'] = torch.cuda.is_bf16_supported()\n"
        "        info['default_device'] = 'cuda:0'\n"
        "    if not devs and hasattr(torch.backends,'mps') and torch.backends.mps.is_available():\n"
        "        devs.append('mps')\n"
        "    devs.append('cpu')\n"
        "    info['devices'] = devs\n"
        "except Exception as e:\n"
        "    info['error'] = str(e)\n"
        "print(json.dumps(info))"
    )
    try:
        result = subprocess.run(
            [PYTHON, "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        import json
        return json.loads(result.stdout.strip())
    except Exception:
        return {
            "torch_version": "?", "cuda_available": False, "cuda_version": None,
            "gpus": [], "bf16": False, "default_device": "cpu",
            "devices": ["cpu"], "error": "detection failed",
        }


HW_INFO = _detect_hw_subprocess()
AVAILABLE_DEVICES = HW_INFO.get("devices", ["cpu"])


def _make_device_combo() -> QComboBox:
    """Create a QComboBox pre-filled with detected devices, best first."""
    cb = QComboBox()
    for dev in AVAILABLE_DEVICES:
        label = dev
        # Annotate GPU devices with their name
        if dev.startswith("cuda:"):
            idx = int(dev.split(":")[1])
            for g in HW_INFO["gpus"]:
                if g["index"] == idx:
                    label = f"{dev}  —  {g['name']} ({g['mem_gb']} GB)"
        cb.addItem(label, dev)  # display text, user data = raw device string
    return cb


def _device_from_combo(cb: QComboBox) -> str:
    """Extract the raw device string from a device combo box."""
    return cb.currentData() or cb.currentText().split()[0]


# ═══════════════════════════════════════════════════════════════════════════
# Worker thread — runs subprocess commands and streams stdout/stderr
# ═══════════════════════════════════════════════════════════════════════════
class WorkerThread(QThread):
    """Run a shell command in a background thread, emitting output line-by-line."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)   # exit code

    def __init__(self, cmd: list[str], cwd: str | Path = ROOT, env: dict | None = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = str(cwd)
        self.env = env or {**os.environ}
        self._proc: Optional[subprocess.Popen] = None

    def run(self):
        self.log_signal.emit(f"▶ {' '.join(self.cmd)}\n")
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=self.env,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            for line in self._proc.stdout:
                self.log_signal.emit(line)
            self._proc.wait()
            code = self._proc.returncode
        except Exception as exc:
            self.log_signal.emit(f"❌ Error: {exc}\n")
            code = -1
        self.finished_signal.emit(code)

    def kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# ═══════════════════════════════════════════════════════════════════════════
# Utility: styled group box
# ═══════════════════════════════════════════════════════════════════════════
def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet("""
        QGroupBox {
            font-weight: 600;
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 14px;
            padding-top: 18px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
    """)
    return g


def _path_row(label_text: str, default: str = "", dialog: str = "file") -> tuple[QLabel, QLineEdit, QPushButton]:
    """Return (label, line_edit, browse_button) for a file/dir path field."""
    lbl = QLabel(label_text)
    le = QLineEdit(default)
    le.setMinimumWidth(260)
    btn = QPushButton("…")
    btn.setFixedWidth(32)
    if dialog == "file":
        btn.clicked.connect(lambda: _browse_file(le))
    else:
        btn.clicked.connect(lambda: _browse_dir(le))
    return lbl, le, btn


def _browse_file(le: QLineEdit):
    path, _ = QFileDialog.getOpenFileName(None, "Select file", str(ROOT))
    if path:
        le.setText(os.path.relpath(path, ROOT))


def _browse_dir(le: QLineEdit):
    path = QFileDialog.getExistingDirectory(None, "Select folder", str(ROOT))
    if path:
        le.setText(os.path.relpath(path, ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation Tab
# ═══════════════════════════════════════════════════════════════════════════
class DataTab(QWidget):
    run_requested = pyqtSignal(list)   # emits the command list

    def __init__(self):
        super().__init__()
        main = QVBoxLayout(self)

        # ── Paths ──
        pg = _group("Paths")
        pf = QGridLayout()
        self.train_csv = self._add_path(pf, 0, "Train CSV:", "data/raw/train.csv")
        self.test_csv = self._add_path(pf, 1, "Test CSV:", "data/raw/test.csv")
        self.augmented_csv = self._add_path(pf, 2, "Augmented CSV:", "data/augmented/augmented_pairs.csv")
        self.lexicon_csv = self._add_path(pf, 3, "Lexicon CSV:", "data/raw/OA_Lexicon_eBL.csv")
        self.processed_dir = self._add_path(pf, 4, "Output Dir:", "data/processed", dialog="dir")
        pg.setLayout(pf)
        main.addWidget(pg)

        # ── Normalization ──
        ng = _group("Normalization")
        nf = QVBoxLayout()
        self.norm_enabled = QCheckBox("Enable normalization")
        self.norm_enabled.setChecked(True)
        self.norm_lowercase = QCheckBox("Lowercase")
        self.norm_whitespace = QCheckBox("Normalize whitespace")
        self.norm_whitespace.setChecked(True)
        self.norm_unicode = QCheckBox("Normalize Unicode punctuation")
        self.norm_unicode.setChecked(True)
        self.norm_separators = QCheckBox("Space repeated separators")
        self.norm_separators.setChecked(True)
        self.norm_damage = QCheckBox("Preserve damage markers")
        self.norm_damage.setChecked(True)
        for cb in (self.norm_enabled, self.norm_lowercase, self.norm_whitespace,
                   self.norm_unicode, self.norm_separators, self.norm_damage):
            nf.addWidget(cb)
        ng.setLayout(nf)
        main.addWidget(ng)

        # ── Splits ──
        sg = _group("Split Configuration")
        sf = QFormLayout()
        self.n_splits = QSpinBox(); self.n_splits.setRange(2, 20); self.n_splits.setValue(5)
        self.split_seed = QSpinBox(); self.split_seed.setRange(0, 99999); self.split_seed.setValue(17)
        self.split_shuffle = QCheckBox(); self.split_shuffle.setChecked(True)
        sf.addRow("K-Fold splits:", self.n_splits)
        sf.addRow("Random seed:", self.split_seed)
        sf.addRow("Shuffle:", self.split_shuffle)
        sg.setLayout(sf)
        main.addWidget(sg)

        # ── View Packing ──
        vg = _group("View Packing")
        vf = QVBoxLayout()
        self.pack_metadata = QCheckBox("Include metadata"); self.pack_metadata.setChecked(True)
        self.pack_raw = QCheckBox("Include raw view"); self.pack_raw.setChecked(True)
        self.pack_norm = QCheckBox("Include normalized view"); self.pack_norm.setChecked(True)
        self.pack_wrap = QCheckBox("Wrap views in tags"); self.pack_wrap.setChecked(True)
        for cb in (self.pack_metadata, self.pack_raw, self.pack_norm, self.pack_wrap):
            vf.addWidget(cb)
        vg.setLayout(vf)
        main.addWidget(vg)

        # ── Config overrides ──
        cg = _group("Config Files (override)")
        cf = QGridLayout()
        self.data_config = self._add_path(cf, 0, "Data config:", DEFAULT_DATA_CFG)
        self.view_config = self._add_path(cf, 1, "View config:", DEFAULT_VIEW_CFG)
        cg.setLayout(cf)
        main.addWidget(cg)

        # ── Run button ──
        self.run_btn = QPushButton("  Prepare Data")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.run_btn.clicked.connect(self._on_run)
        main.addWidget(self.run_btn)

        main.addStretch()

    def _add_path(self, grid: QGridLayout, row: int, label: str, default: str, dialog: str = "file") -> QLineEdit:
        lbl, le, btn = _path_row(label, default, dialog)
        grid.addWidget(lbl, row, 0)
        grid.addWidget(le, row, 1)
        grid.addWidget(btn, row, 2)
        return le

    def _on_run(self):
        cmd = [
            PYTHON, "-m", "src.data.prepare",
            "--data-config", self.data_config.text(),
            "--view-config", self.view_config.text(),
        ]
        self.run_requested.emit(cmd)


# ═══════════════════════════════════════════════════════════════════════════
# Training Tab
# ═══════════════════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    run_requested = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        main = QVBoxLayout(self)

        # ── Run info ──
        rg = _group("Run")
        rf = QFormLayout()
        self.run_name = QLineEdit("hybrid_base")
        self.output_dir = QLineEdit("outputs/runs/hybrid_base")
        self.seed = QSpinBox(); self.seed.setRange(0, 99999); self.seed.setValue(17)
        self.fold = QSpinBox(); self.fold.setRange(0, 19); self.fold.setValue(0)
        self.device = _make_device_combo()
        self.use_bf16 = QCheckBox("Use bf16 (RTX 30xx+)")
        self.use_bf16.setChecked(HW_INFO["bf16"])
        self.use_bf16.setEnabled(HW_INFO["bf16"])
        rf.addRow("Run name:", self.run_name)
        rf.addRow("Output dir:", self.output_dir)
        rf.addRow("Seed:", self.seed)
        rf.addRow("Fold:", self.fold)
        rf.addRow("Device:", self.device)
        rf.addRow("Precision:", self.use_bf16)
        rg.setLayout(rf)
        main.addWidget(rg)

        # ── Hyperparameters ──
        hg = _group("Hyperparameters")
        hf = QFormLayout()
        self.epochs = QSpinBox(); self.epochs.setRange(1, 200); self.epochs.setValue(15)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 512); self.batch_size.setValue(16)
        self.eval_batch_size = QSpinBox(); self.eval_batch_size.setRange(1, 512); self.eval_batch_size.setValue(16)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-7, 1.0)
        self.lr.setValue(3e-4); self.lr.setSingleStep(1e-5)
        self.weight_decay = QDoubleSpinBox(); self.weight_decay.setDecimals(4)
        self.weight_decay.setRange(0, 1); self.weight_decay.setValue(0.01)
        self.warmup_steps = QSpinBox(); self.warmup_steps.setRange(0, 50000); self.warmup_steps.setValue(500)
        self.grad_clip = QDoubleSpinBox(); self.grad_clip.setDecimals(2)
        self.grad_clip.setRange(0, 100); self.grad_clip.setValue(1.0)
        hf.addRow("Epochs:", self.epochs)
        hf.addRow("Batch size:", self.batch_size)
        hf.addRow("Eval batch size:", self.eval_batch_size)
        hf.addRow("Learning rate:", self.lr)
        hf.addRow("Weight decay:", self.weight_decay)
        hf.addRow("Warmup steps:", self.warmup_steps)
        hf.addRow("Gradient clip norm:", self.grad_clip)
        rg2 = _group("Logging / Checkpointing")
        rf2 = QFormLayout()
        self.log_every = QSpinBox(); self.log_every.setRange(1, 10000); self.log_every.setValue(20)
        self.eval_every = QSpinBox(); self.eval_every.setRange(1, 50000); self.eval_every.setValue(250)
        self.ckpt_every = QSpinBox(); self.ckpt_every.setRange(1, 50000); self.ckpt_every.setValue(500)
        rf2.addRow("Log every N steps:", self.log_every)
        rf2.addRow("Eval every N steps:", self.eval_every)
        rf2.addRow("Checkpoint every N steps:", self.ckpt_every)
        rg2.setLayout(rf2)
        hg.setLayout(hf)
        main.addWidget(hg)
        main.addWidget(rg2)

        # ── Model Architecture ──
        mg = _group("Model Architecture")
        mf = QFormLayout()

        self.enc_d_model = QSpinBox(); self.enc_d_model.setRange(64, 2048); self.enc_d_model.setValue(512)
        self.enc_layers = QSpinBox(); self.enc_layers.setRange(1, 32); self.enc_layers.setValue(8)
        self.enc_dropout = QDoubleSpinBox(); self.enc_dropout.setDecimals(2)
        self.enc_dropout.setRange(0, 0.9); self.enc_dropout.setValue(0.1)
        self.enc_bidir = QCheckBox(); self.enc_bidir.setChecked(True)
        self.enc_mamba = QCheckBox(); self.enc_mamba.setChecked(True)

        self.dec_d_model = QSpinBox(); self.dec_d_model.setRange(64, 2048); self.dec_d_model.setValue(512)
        self.dec_layers = QSpinBox(); self.dec_layers.setRange(1, 32); self.dec_layers.setValue(4)
        self.dec_heads = QSpinBox(); self.dec_heads.setRange(1, 64); self.dec_heads.setValue(8)
        self.dec_ff_mult = QSpinBox(); self.dec_ff_mult.setRange(1, 16); self.dec_ff_mult.setValue(4)
        self.dec_dropout = QDoubleSpinBox(); self.dec_dropout.setDecimals(2)
        self.dec_dropout.setRange(0, 0.9); self.dec_dropout.setValue(0.1)

        mf.addRow("Encoder d_model:", self.enc_d_model)
        mf.addRow("Encoder layers:", self.enc_layers)
        mf.addRow("Encoder dropout:", self.enc_dropout)
        mf.addRow("Bidirectional:", self.enc_bidir)
        mf.addRow("Use Mamba SSM:", self.enc_mamba)
        mf.addRow(QFrame())  # spacer
        mf.addRow("Decoder d_model:", self.dec_d_model)
        mf.addRow("Decoder layers:", self.dec_layers)
        mf.addRow("Decoder heads:", self.dec_heads)
        mf.addRow("Decoder FF mult:", self.dec_ff_mult)
        mf.addRow("Decoder dropout:", self.dec_dropout)
        mg.setLayout(mf)
        main.addWidget(mg)

        # ── Input / tokenization ──
        ig = _group("Input / Tokenization")
        iff = QFormLayout()
        self.src_max_len = QSpinBox(); self.src_max_len.setRange(64, 8192); self.src_max_len.setValue(1024)
        self.tgt_max_len = QSpinBox(); self.tgt_max_len.setRange(64, 4096); self.tgt_max_len.setValue(256)
        self.target_tokenizer = QLineEdit("t5-small")
        self.include_metadata = QCheckBox(); self.include_metadata.setChecked(True)
        self.include_raw = QCheckBox(); self.include_raw.setChecked(True)
        self.include_norm = QCheckBox(); self.include_norm.setChecked(True)
        iff.addRow("Source max length:", self.src_max_len)
        iff.addRow("Target max length:", self.tgt_max_len)
        iff.addRow("Target tokenizer:", self.target_tokenizer)
        iff.addRow("Include metadata:", self.include_metadata)
        iff.addRow("Include raw view:", self.include_raw)
        iff.addRow("Include normalized:", self.include_norm)
        ig.setLayout(iff)
        main.addWidget(ig)

        # ── Auxiliary losses ──
        ag = _group("Auxiliary Loss Weights")
        af = QFormLayout()
        self.name_w = QDoubleSpinBox(); self.name_w.setDecimals(2); self.name_w.setRange(0, 10); self.name_w.setValue(0.2)
        self.number_w = QDoubleSpinBox(); self.number_w.setDecimals(2); self.number_w.setRange(0, 10); self.number_w.setValue(0.1)
        self.damage_w = QDoubleSpinBox(); self.damage_w.setDecimals(2); self.damage_w.setRange(0, 10); self.damage_w.setValue(0.1)
        af.addRow("Name weight:", self.name_w)
        af.addRow("Number weight:", self.number_w)
        af.addRow("Damage weight:", self.damage_w)
        ag.setLayout(af)
        main.addWidget(ag)

        # ── Config overrides ──
        cg = _group("Config Files (override)")
        cf = QGridLayout()
        self.data_config = self._add_path(cf, 0, "Data config:", DEFAULT_DATA_CFG)
        self.view_config = self._add_path(cf, 1, "View config:", DEFAULT_VIEW_CFG)
        self.model_config = self._add_path(cf, 2, "Model config:", DEFAULT_MODEL_CFG)
        self.train_config = self._add_path(cf, 3, "Train config:", DEFAULT_TRAIN_CFG)
        cg.setLayout(cf)
        main.addWidget(cg)

        # ── Run / Stop ──
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("  Start Training")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("  Stop")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-size: 13px;")
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        main.addLayout(btn_row)

        main.addStretch()

    def _add_path(self, grid: QGridLayout, row: int, label: str, default: str) -> QLineEdit:
        lbl, le, btn = _path_row(label, default)
        grid.addWidget(lbl, row, 0)
        grid.addWidget(le, row, 1)
        grid.addWidget(btn, row, 2)
        return le

    def _on_run(self):
        cmd = [
            PYTHON, "-m", "src.train.train_hybrid",
            "--data-config", self.data_config.text(),
            "--view-config", self.view_config.text(),
            "--model-config", self.model_config.text(),
            "--train-config", self.train_config.text(),
            "--device", _device_from_combo(self.device),
        ]
        if self.use_bf16.isChecked():
            cmd.append("--bf16")
        self.run_requested.emit(cmd)


# ═══════════════════════════════════════════════════════════════════════════
# Inference Tab
# ═══════════════════════════════════════════════════════════════════════════
class InferenceTab(QWidget):
    run_requested = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        main = QVBoxLayout(self)

        # ── Checkpoint + I/O ──
        ig = _group("Input / Output")
        iff = QGridLayout()
        self.checkpoint = self._add_path(iff, 0, "Checkpoint (.pt):", "outputs/runs/hybrid_base/best.pt")
        self.output_csv = self._add_path(iff, 1, "Output CSV:", "submission.csv")
        ig.setLayout(iff)
        main.addWidget(ig)

        # ── Decoding params ──
        dg = _group("Decoding Parameters")
        df = QFormLayout()
        self.num_beams = QSpinBox(); self.num_beams.setRange(1, 64); self.num_beams.setValue(5)
        self.length_penalty = QDoubleSpinBox(); self.length_penalty.setDecimals(2)
        self.length_penalty.setRange(0, 5); self.length_penalty.setValue(0.9)
        self.no_repeat_ngram = QSpinBox(); self.no_repeat_ngram.setRange(0, 20); self.no_repeat_ngram.setValue(3)
        self.max_new_tokens = QSpinBox(); self.max_new_tokens.setRange(16, 2048); self.max_new_tokens.setValue(256)
        self.inf_batch_size = QSpinBox(); self.inf_batch_size.setRange(1, 512); self.inf_batch_size.setValue(8)
        self.inf_device = _make_device_combo()
        self.inf_bf16 = QCheckBox("Use bf16"); self.inf_bf16.setChecked(HW_INFO["bf16"]); self.inf_bf16.setEnabled(HW_INFO["bf16"])
        df.addRow("Num beams:", self.num_beams)
        df.addRow("Length penalty:", self.length_penalty)
        df.addRow("No-repeat n-gram:", self.no_repeat_ngram)
        df.addRow("Max new tokens:", self.max_new_tokens)
        df.addRow("Batch size:", self.inf_batch_size)
        df.addRow("Device:", self.inf_device)
        df.addRow("Precision:", self.inf_bf16)
        dg.setLayout(df)
        main.addWidget(dg)

        # ── Config overrides ──
        cg = _group("Config Files (override)")
        cf = QGridLayout()
        self.data_config = self._add_path(cf, 0, "Data config:", DEFAULT_DATA_CFG)
        self.view_config = self._add_path(cf, 1, "View config:", DEFAULT_VIEW_CFG)
        self.model_config = self._add_path(cf, 2, "Model config:", DEFAULT_MODEL_CFG)
        cg.setLayout(cf)
        main.addWidget(cg)

        # ── Run ──
        self.run_btn = QPushButton("  Generate Submission")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.run_btn.clicked.connect(self._on_run)
        main.addWidget(self.run_btn)

        # ── Quick translate ──
        tg = _group("Quick Translate (single sentence)")
        tf = QVBoxLayout()
        self.quick_input = QTextEdit()
        self.quick_input.setPlaceholderText("Paste Akkadian transliteration here…")
        self.quick_input.setMaximumHeight(80)
        self.quick_output = QTextEdit()
        self.quick_output.setPlaceholderText("English translation will appear here…")
        self.quick_output.setMaximumHeight(80)
        self.quick_output.setReadOnly(True)
        self.quick_btn = QPushButton("Translate")
        self.quick_btn.clicked.connect(self._on_quick_translate)
        tf.addWidget(QLabel("Source:"))
        tf.addWidget(self.quick_input)
        tf.addWidget(self.quick_btn)
        tf.addWidget(QLabel("Translation:"))
        tf.addWidget(self.quick_output)
        tg.setLayout(tf)
        main.addWidget(tg)

        main.addStretch()

    def _add_path(self, grid: QGridLayout, row: int, label: str, default: str) -> QLineEdit:
        lbl, le, btn = _path_row(label, default)
        grid.addWidget(lbl, row, 0)
        grid.addWidget(le, row, 1)
        grid.addWidget(btn, row, 2)
        return le

    def _on_run(self):
        cmd = [
            PYTHON, "-m", "src.eval.decode",
            "--data-config", self.data_config.text(),
            "--view-config", self.view_config.text(),
            "--model-config", self.model_config.text(),
            "--checkpoint", self.checkpoint.text(),
            "--output", self.output_csv.text(),
            "--batch-size", str(self.inf_batch_size.value()),
            "--device", _device_from_combo(self.inf_device),
            "--num-beams", str(self.num_beams.value()),
        ]
        self.run_requested.emit(cmd)

    def _on_quick_translate(self):
        """Quick single-sentence translation using a lightweight subprocess."""
        source = self.quick_input.toPlainText().strip()
        if not source:
            return
        # Build a tiny Python snippet that loads model and translates one sentence
        script = (
            "import torch, sys; sys.path.insert(0,'.'); "
            "from src.eval.decode import load_model, decode_batch, preprocess_test_row; "
            "from src.data.collators import ByteSourceEncoder; "
            "from src.utils.io import load_yaml; "
            f"model_cfg = load_yaml('{self.model_config.text()}'); "
            f"data_cfg = load_yaml('{self.data_config.text()}'); "
            f"view_cfg = load_yaml('{self.view_config.text()}'); "
            f"dev = torch.device('{_device_from_combo(self.inf_device)}' if torch.cuda.is_available() else 'cpu'); "
            f"model = load_model(model_cfg, '{self.checkpoint.text()}', dev); "
            f"se = ByteSourceEncoder(max_length=model_cfg['input']['source_max_length']); "
            f"row = {{'transliteration': '''{source}'''}}; "
            "packed = preprocess_test_row(row, data_cfg.get('normalization',{}), view_cfg.get('packing',{})); "
            f"preds = decode_batch(model, se, [packed], dev, num_beams={self.num_beams.value()}); "
            "print(preds[0])"
        )
        cmd = [PYTHON, "-c", script]
        self.quick_output.setPlainText("Translating…")
        self.run_requested.emit(cmd)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Tab
# ═══════════════════════════════════════════════════════════════════════════
class EvalTab(QWidget):
    run_requested = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        main = QVBoxLayout(self)

        # ── Predictions input ──
        pg = _group("Evaluate Predictions")
        pf = QGridLayout()
        self.pred_csv = self._add_path(pf, 0, "Predictions CSV:", "outputs/runs/hybrid_base/val_predictions.csv")
        pg.setLayout(pf)
        main.addWidget(pg)

        # ── Metrics selection ──
        mg = _group("Metrics")
        mf = QVBoxLayout()
        self.m_bleu = QCheckBox("SacreBLEU"); self.m_bleu.setChecked(True)
        self.m_chrf = QCheckBox("chrF++"); self.m_chrf.setChecked(True)
        self.m_comp = QCheckBox("Competition score √(BLEU×chrF++)"); self.m_comp.setChecked(True)
        self.m_name = QCheckBox("Name span F1"); self.m_name.setChecked(True)
        self.m_number = QCheckBox("Number exact match"); self.m_number.setChecked(True)
        self.m_damage = QCheckBox("Damage hallucination proxy"); self.m_damage.setChecked(True)
        self.m_lenratio = QCheckBox("Length ratio"); self.m_lenratio.setChecked(True)
        for cb in (self.m_bleu, self.m_chrf, self.m_comp, self.m_name,
                   self.m_number, self.m_damage, self.m_lenratio):
            mf.addWidget(cb)
        mg.setLayout(mf)
        main.addWidget(mg)

        # ── Error analysis ──
        eg = _group("Error Analysis")
        ef = QGridLayout()
        self.error_output = self._add_path(ef, 0, "Error buckets output:", "outputs/runs/hybrid_base/error_buckets.csv")
        eg.setLayout(ef)
        main.addWidget(eg)

        # ── Run ──
        self.run_btn = QPushButton("  Run Evaluation")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.run_btn.clicked.connect(self._on_run)
        main.addWidget(self.run_btn)

        main.addStretch()

    def _add_path(self, grid: QGridLayout, row: int, label: str, default: str) -> QLineEdit:
        lbl, le, btn = _path_row(label, default)
        grid.addWidget(lbl, row, 0)
        grid.addWidget(le, row, 1)
        grid.addWidget(btn, row, 2)
        return le

    def _on_run(self):
        pred_csv = self.pred_csv.text()
        script = (
            "import pandas as pd, json, sys; sys.path.insert(0,'.'); "
            "from src.eval.metrics import all_metrics; "
            "from src.eval.error_buckets import build_error_buckets; "
            f"df = pd.read_csv('{pred_csv}'); "
            "metrics = all_metrics(df['prediction'].tolist(), df['reference'].tolist()); "
            "print(json.dumps(metrics, indent=2)); "
            f"errors = build_error_buckets(df); "
            f"errors.to_csv('{self.error_output.text()}', index=False); "
            f"print(f'\\nWrote error buckets to {self.error_output.text()}')"
        )
        cmd = [PYTHON, "-c", script]
        self.run_requested.emit(cmd)


# ═══════════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════════
class MardukGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARDUK — Mamba-Augmented Reconstruction & Decoding of Unknown Kuneiform")
        self.setMinimumSize(880, 700)
        self.resize(960, 820)

        self._worker: Optional[WorkerThread] = None

        # ── Central widget ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Header ──
        hdr = QLabel("𒀭  MARDUK  𒀭")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("font-size: 20px; font-weight: bold; padding: 6px;")
        root.addWidget(hdr)

        # ── Hardware info banner ──
        hw_parts = [f"PyTorch {HW_INFO['torch_version']}"]
        if HW_INFO["cuda_available"]:
            for g in HW_INFO["gpus"]:
                hw_parts.append(f"GPU {g['index']}: {g['name']} ({g['mem_gb']} GB)")
            hw_parts.append(f"CUDA {HW_INFO['cuda_version']}")
            if HW_INFO["bf16"]:
                hw_parts.append("bf16 ✓")
            banner_color = "#264f36"  # green-ish = GPU available
            banner_icon = "🟢"
        else:
            hw_parts.append("NO GPU DETECTED — running on CPU")
            banner_color = "#6b3030"  # red-ish = CPU only
            banner_icon = "🔴"
        hw_banner = QLabel(f"  {banner_icon}  {'  |  '.join(hw_parts)}")
        hw_banner.setAlignment(Qt.AlignCenter)
        hw_banner.setStyleSheet(f"""
            background-color: {banner_color};
            color: #e0e0e0;
            border-radius: 4px;
            padding: 5px;
            font-size: 11px;
        """)
        root.addWidget(hw_banner)

        # ── Splitter: tabs on top, console on bottom ──
        splitter = QSplitter(Qt.Vertical)

        # ── Tabs ──
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        from PyQt5.QtWidgets import QScrollArea

        self.data_tab = DataTab()
        self.train_tab = TrainTab()
        self.inference_tab = InferenceTab()
        self.eval_tab = EvalTab()

        for tab, title in [
            (self.data_tab, " Data Prep"),
            (self.train_tab, " Train"),
            (self.inference_tab, " Inference"),
            (self.eval_tab, " Evaluate"),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab)
            self.tabs.addTab(scroll, title)

        splitter.addWidget(self.tabs)

        # ── Console / log ──
        console_frame = QWidget()
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(0, 0, 0, 0)

        console_header = QHBoxLayout()
        console_label = QLabel("Console")
        console_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(lambda: self.console.clear())
        self.kill_btn = QPushButton("Kill Process")
        self.kill_btn.setFixedWidth(90)
        self.kill_btn.setEnabled(False)
        self.kill_btn.clicked.connect(self._kill_worker)
        console_header.addWidget(console_label)
        console_header.addStretch()
        console_header.addWidget(self.kill_btn)
        console_header.addWidget(self.clear_btn)
        console_layout.addLayout(console_header)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 9))
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        console_layout.addWidget(self.console)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        console_layout.addWidget(self.progress)

        splitter.addWidget(console_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # ── Wire signals ──
        self.data_tab.run_requested.connect(self._run_command)
        self.train_tab.run_requested.connect(self._run_command)
        self.train_tab.stop_btn.clicked.connect(self._kill_worker)
        self.inference_tab.run_requested.connect(self._run_command)
        self.eval_tab.run_requested.connect(self._run_command)

        self._apply_dark_theme()

    # ── Command execution ──
    def _run_command(self, cmd: list[str]):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A process is already running. Stop it first.")
            return
        self.console.append("")
        self._set_running(True)
        self._worker = WorkerThread(cmd, cwd=ROOT)
        self._worker.log_signal.connect(self._on_log)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_log(self, text: str):
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.End)

    def _on_finished(self, code: int):
        color = "#4ec9b0" if code == 0 else "#f44747"
        self.console.append(f'<span style="color:{color};">✦ Process exited with code {code}</span>\n')
        self._set_running(False)

    def _kill_worker(self):
        if self._worker:
            self._worker.kill()
            self.console.append('<span style="color:#ce9178;">⚠ Process terminated by user</span>\n')

    def _set_running(self, running: bool):
        self.progress.setVisible(running)
        self.kill_btn.setEnabled(running)
        self.train_tab.stop_btn.setEnabled(running)
        for tab in (self.data_tab, self.train_tab, self.inference_tab, self.eval_tab):
            tab.run_btn.setEnabled(not running)

    # ── Dark theme ──
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #252526;
                color: #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: #cccccc;
                padding: 8px 18px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #3a3a3a;
            }
            QGroupBox {
                color: #4ec9b0;
                font-size: 12px;
            }
            QLabel {
                color: #cccccc;
                font-size: 11px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #666;
                border-radius: 3px;
                background: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background: #0078d4;
                border-color: #0078d4;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 6px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666666;
            }
            QProgressBar {
                border: none;
                background: #2d2d2d;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #0078d4;
                border-radius: 3px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QTextEdit {
                font-size: 11px;
            }
        """)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MARDUK")
    app.setStyle("Fusion")  # cross-platform modern look

    window = MardukGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
