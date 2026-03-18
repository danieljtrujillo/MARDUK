"""Prepare training data in plain prefix format (compatible with public ByT5 models).

This creates train_prepared_plain.csv with:
- packed_source = "translate Akkadian to English: " + preprocessed transliteration
- target_text = cleaned translation
- ONLY train.csv data (no augmented data to avoid poisoning)

Usage:
    python -m src.data.prepare_plain --data-config configs/data/raw.yaml
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

from src.data.normalize import clean_translation
from src.data.splitters import FoldConfig, add_group_kfold_column
from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import get_logger


# ── Preprocessing matching public ByT5 models ──

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú",
                         "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù",
                         "A": "À", "E": "È", "I": "Ì", "U": "Ù"})


def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)

_CHAR_TRANS = str.maketrans({
    "ḫ": "h", "Ḫ": "H", "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
})

_DET_UPPER_RE = re.compile(
    r"\(([A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF0-9]{1,6})\)"
)
_DET_LOWER_RE = re.compile(
    r"\(([a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF]{1,4})\)"
)

_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_WS_RE = re.compile(r"\s+")

_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}


def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]


def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")


def preprocess_transliteration(text: str) -> str:
    """Preprocess transliteration matching public ByT5 model training format."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _ascii_to_diacritics(text)
    text = _DET_UPPER_RE.sub(r"\1", text)
    text = _DET_LOWER_RE.sub(r"{\1}", text)
    text = _GAP_UNIFIED_RE.sub("<gap>", text)
    text = text.translate(_CHAR_TRANS)
    text = text.replace("ₓ", "")
    text = _KUBABBAR_RE.sub("KÙ.BABBAR", text)
    text = _EXACT_FRAC_RE.sub(_frac_repl, text)
    text = _FLOAT_RE.sub(lambda m: _canon_decimal(float(m.group(1))), text)
    text = _WS_RE.sub(" ", text).strip()
    return "translate Akkadian to English: " + text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger()

    data_cfg = load_yaml(args.data_config)
    processed_dir = ensure_dir(data_cfg["paths"]["processed_dir"])

    # ── Load train.csv ONLY (no augmented data) ──
    train_path = data_cfg["paths"]["train_csv"]
    logger.info("Loading train.csv from %s", train_path)
    train_df = pd.read_csv(train_path)

    id_col = data_cfg["columns"].get("id", "oare_id")
    src_col = data_cfg["columns"]["source"]
    tgt_col = data_cfg["columns"].get("target", "translation")

    # ── Fix truncated transliterations using published_texts ──
    pt_path = data_cfg["paths"].get("published_texts_csv")
    if pt_path and Path(pt_path).exists():
        pt_df = pd.read_csv(pt_path)
        if id_col in pt_df.columns and src_col in pt_df.columns:
            pt_map = pt_df.set_index(id_col)[src_col].dropna().to_dict()
            n_fixed = 0
            for idx, row in train_df.iterrows():
                rid = row[id_col]
                if rid in pt_map:
                    full = pt_map[rid]
                    cur = str(row[src_col])
                    if len(full) > len(cur) + 10:
                        train_df.at[idx, src_col] = full
                        n_fixed += 1
            if n_fixed:
                logger.info("Fixed %d truncated transliterations", n_fixed)

    logger.info("Train samples: %d", len(train_df))

    # ── Build packed_source with plain prefix format ──
    train_df["packed_source"] = train_df[src_col].map(preprocess_transliteration)

    # ── Clean targets ──
    train_df["target_text"] = train_df[tgt_col].fillna("").map(clean_translation)

    # ── Set text_id (for GroupKFold and to avoid tablet leakage) ──
    if "text_id" not in train_df.columns:
        # Use oare_id prefix as group
        train_df["text_id"] = train_df[id_col].astype(str)

    # ── Drop empty targets ──
    n_before = len(train_df)
    train_df = train_df[train_df["target_text"].str.strip().astype(bool)].reset_index(drop=True)
    n_dropped = n_before - len(train_df)
    if n_dropped:
        logger.info("Dropped %d rows with empty target_text", n_dropped)

    # ── Add fold column ──
    split_cfg = data_cfg["splits"]
    folds = FoldConfig(
        n_splits=split_cfg["n_splits"],
        random_state=split_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
    )
    train_df = add_group_kfold_column(train_df, folds, group_col="text_id")

    # ── Save ──
    out_path = processed_dir / "train_prepared_plain.csv"
    # Keep only needed columns
    keep_cols = ["packed_source", "target_text", "text_id", "fold"]
    if id_col in train_df.columns:
        keep_cols.insert(0, id_col)
    train_df[keep_cols].to_csv(out_path, index=False)

    logger.info("Wrote %s (%d rows)", out_path, len(train_df))
    for fold_id in sorted(train_df["fold"].unique()):
        fold_size = len(train_df[train_df["fold"] == fold_id])
        logger.info("  Fold %d: %d examples", fold_id, fold_size)

    # Print sample
    logger.info("Sample packed_source: %s", train_df["packed_source"].iloc[0][:200])
    logger.info("Sample target_text: %s", train_df["target_text"].iloc[0][:200])


if __name__ == "__main__":
    main()
