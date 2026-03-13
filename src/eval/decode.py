"""Full inference / decoding pipeline for Kaggle submission.

Usage (local testing):
    python -m src.eval.decode \
        --data-config configs/data/raw.yaml \
        --view-config configs/data/dual_view.yaml \
        --model-config configs/model/mamba_enc_txd_dec_base.yaml \
        --checkpoint outputs/runs/hybrid_base/best.pt \
        --output submission.csv

For Kaggle notebook submission, call `generate_submission()` directly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.build_dual_view import build_metadata_prefix, pack_source
from src.data.collators import ByteSourceEncoder
from src.data.normalize import normalize_text
from src.models.hybrid_seq2seq import HybridSeq2Seq
from src.utils.io import load_yaml
from src.utils.logging import get_logger


def strip_special(text: str) -> str:
    """Remove common special tokens from decoded text."""
    for tok in ["<pad>", "</s>", "<s>", "<unk>"]:
        text = text.replace(tok, " ")
    return " ".join(text.split()).strip()


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


def load_model(
    model_cfg: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> HybridSeq2Seq:
    """Load a trained HybridSeq2Seq model from checkpoint."""
    source_encoder = ByteSourceEncoder(max_length=model_cfg["input"]["source_max_length"])
    model = HybridSeq2Seq(
        source_vocab_size=source_encoder.vocab_size,
        target_tokenizer_name_or_path=model_cfg["target_tokenizer_name_or_path"],
        encoder_cfg=model_cfg["encoder"],
        decoder_cfg=model_cfg["decoder"],
        aux_weights=model_cfg.get("auxiliary_losses"),
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def decode_batch(
    model: HybridSeq2Seq,
    source_encoder: ByteSourceEncoder,
    packed_sources: list[str],
    device: torch.device,
    max_new_tokens: int = 256,
    num_beams: int = 5,
    length_penalty: float = 0.9,
    no_repeat_ngram_size: int = 3,
) -> list[str]:
    """Encode a batch of packed source strings and decode translations."""
    source_ids = [source_encoder.encode(s) for s in packed_sources]
    source_ids_t, source_mask_t = source_encoder.pad_batch(source_ids)
    source_ids_t = source_ids_t.to(device)
    source_mask_t = source_mask_t.to(device)

    generated = model.generate(
        source_ids=source_ids_t,
        source_mask=source_mask_t,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    tokenizer = model.target_tokenizer
    predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [strip_special(p) for p in predictions]


def generate_submission(
    test_csv_path: str | Path,
    model_cfg: dict,
    data_cfg: dict,
    view_cfg: dict,
    checkpoint_path: str | Path,
    output_path: str | Path = "submission.csv",
    batch_size: int = 8,
    device: str = "cuda",
    num_beams: int = 5,
) -> pd.DataFrame:
    """Generate a Kaggle submission CSV from test data and a trained model.

    Returns the submission DataFrame.
    """
    logger = get_logger()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model from %s", checkpoint_path)
    model = load_model(model_cfg, checkpoint_path, dev)
    source_encoder = ByteSourceEncoder(max_length=model_cfg["input"]["source_max_length"])

    # Load and preprocess test data
    logger.info("Loading test data from %s", test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    norm_cfg = data_cfg.get("normalization", {})
    packing_cfg = view_cfg.get("packing", {})
    metadata_cols = data_cfg.get("columns", {}).get("metadata", [])

    packed_sources = []
    for _, row in test_df.iterrows():
        packed = preprocess_test_row(row.to_dict(), norm_cfg, packing_cfg, metadata_cols)
        packed_sources.append(packed)

    logger.info("Preprocessing complete. %d test examples.", len(packed_sources))

    # Sort by text_id for sibling context (multiple sentences from same tablet)
    if "text_id" in test_df.columns:
        test_df["_packed"] = packed_sources
        test_df = test_df.sort_values(["text_id", "line_start" if "line_start" in test_df.columns else "id"])
        packed_sources = test_df["_packed"].tolist()
        test_df = test_df.drop(columns=["_packed"])

    # Decode in batches
    all_predictions: list[str] = []
    for i in tqdm(range(0, len(packed_sources), batch_size), desc="Decoding"):
        batch = packed_sources[i : i + batch_size]
        preds = decode_batch(
            model=model,
            source_encoder=source_encoder,
            packed_sources=batch,
            device=dev,
            max_new_tokens=model_cfg["input"].get("target_max_length", 256),
            num_beams=num_beams,
            length_penalty=0.9,
            no_repeat_ngram_size=3,
        )
        all_predictions.extend(preds)

    # Build submission
    submission = pd.DataFrame({
        "id": test_df["id"].values if "id" in test_df.columns else range(len(test_df)),
        "translation": all_predictions,
    })

    # Ensure original order by id
    submission = submission.sort_values("id").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Wrote submission to %s (%d rows)", output_path, len(submission))

    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV from trained hybrid model")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--view-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-beams", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    view_cfg = load_yaml(args.view_config)
    model_cfg = load_yaml(args.model_config)

    test_csv = data_cfg["paths"]["test_csv"]

    generate_submission(
        test_csv_path=test_csv,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        view_cfg=view_cfg,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
