"""Inference / decoding pipeline for ByT5 model → Kaggle submission.

Usage (local testing):
    python -m src.eval.decode_byt5 \
        --data-config configs/data/raw.yaml \
        --view-config configs/data/dual_view.yaml \
        --model-config configs/model/byt5_base.yaml \
        --checkpoint outputs/runs/byt5_base/best \
        --output submission.csv

For Kaggle notebook, call `generate_submission()` directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.build_dual_view import build_metadata_prefix, pack_source
from src.data.normalize import normalize_text
from src.utils.io import load_yaml
from src.utils.logging import get_logger


def preprocess_test_row(
    row: dict,
    norm_cfg: dict,
    packing_cfg: dict,
    metadata_cols: list[str] | None = None,
) -> str:
    """Normalize and pack a single test row's transliteration."""
    source = str(row.get("transliteration", ""))
    normalized = normalize_text(
        source,
        lowercase=norm_cfg.get("lowercase", False),
        normalize_whitespace=norm_cfg.get("normalize_whitespace", True),
        normalize_unicode_punctuation=norm_cfg.get("normalize_unicode_punctuation", True),
        space_repeated_separators=norm_cfg.get("space_repeated_separators", True),
        preserve_damage_markers=norm_cfg.get("preserve_damage_markers", True),
        apply_akkadian_normalization=True,
    )
    metadata_prefix = build_metadata_prefix(row, metadata_cols or [])
    packed = pack_source(
        raw_text=source,
        normalized_text=normalized,
        metadata_prefix=metadata_prefix if packing_cfg.get("include_metadata") else None,
        include_raw_view=packing_cfg.get("include_raw_view", True),
        include_normalized_view=packing_cfg.get("include_normalized_view", True),
        wrap_views=packing_cfg.get("wrap_views", True),
    )
    return packed


@torch.no_grad()
def decode_batch(
    model,
    tokenizer,
    packed_sources: list[str],
    device: torch.device,
    src_max: int = 1024,
    tgt_max: int = 256,
    num_beams: int = 5,
    length_penalty: float = 0.9,
    no_repeat_ngram_size: int = 3,
) -> list[str]:
    """Tokenize a batch and generate translations."""
    inputs = tokenizer(
        packed_sources,
        max_length=src_max,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    device_type = device.type
    use_amp = device_type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        generated = model.generate(
            **inputs,
            max_new_tokens=tgt_max,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def generate_submission(
    test_csv_path: str | Path,
    model_cfg: dict,
    data_cfg: dict,
    view_cfg: dict,
    checkpoint_path: str | Path,
    output_path: str | Path = "submission.csv",
    batch_size: int = 8,
    device: str = "cuda",
) -> pd.DataFrame:
    """Generate a Kaggle submission CSV from test data and a trained ByT5 model."""
    logger = get_logger()

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(dev),
                     torch.cuda.get_device_properties(dev).total_memory / 1024**3)

    src_max = model_cfg["source_max_length"]
    tgt_max = model_cfg["target_max_length"]
    gen_cfg = model_cfg.get("generation", {})

    # Load model
    logger.info("Loading ByT5 from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(dev)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.1fM params", n_params / 1e6)

    # Load and preprocess test data
    logger.info("Loading test data from %s", test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    norm_cfg = data_cfg.get("normalization", {})
    packing_cfg = view_cfg.get("packing", {})
    metadata_cols = data_cfg.get("columns", {}).get("metadata", [])

    packed_sources = [
        preprocess_test_row(row.to_dict(), norm_cfg, packing_cfg, metadata_cols)
        for _, row in test_df.iterrows()
    ]
    logger.info("Preprocessed %d test examples.", len(packed_sources))

    # Sort by text_id for sibling context
    if "text_id" in test_df.columns:
        test_df["_packed"] = packed_sources
        sort_cols = ["text_id"]
        if "line_start" in test_df.columns:
            sort_cols.append("line_start")
        test_df = test_df.sort_values(sort_cols)
        packed_sources = test_df["_packed"].tolist()
        test_df = test_df.drop(columns=["_packed"])

    # Decode in batches
    all_predictions: list[str] = []
    for i in tqdm(range(0, len(packed_sources), batch_size), desc="Decoding"):
        batch = packed_sources[i : i + batch_size]
        preds = decode_batch(
            model=model,
            tokenizer=tokenizer,
            packed_sources=batch,
            device=dev,
            src_max=src_max,
            tgt_max=tgt_max,
            num_beams=gen_cfg.get("num_beams", 5),
            length_penalty=gen_cfg.get("length_penalty", 0.9),
            no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size", 3),
        )
        all_predictions.extend(preds)

    # Build submission
    id_col = "id" if "id" in test_df.columns else None
    submission = pd.DataFrame({
        "id": test_df[id_col].values if id_col else range(len(test_df)),
        "translation": all_predictions,
    })
    submission = submission.sort_values("id").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Wrote submission to %s (%d rows)", output_path, len(submission))

    return submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate submission CSV from trained ByT5")
    p.add_argument("--data-config", required=True)
    p.add_argument("--view-config", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    view_cfg = load_yaml(args.view_config)
    model_cfg = load_yaml(args.model_config)

    generate_submission(
        test_csv_path=data_cfg["paths"]["test_csv"],
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        view_cfg=view_cfg,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
