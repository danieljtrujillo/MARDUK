"""Expand training data by cross-referencing Sentences_Oare with published_texts.

Produces data/augmented/augmented_pairs.csv with columns:
    oare_id, transliteration, translation

Usage:
    python -m src.data.expand_training --data-config configs/data/raw.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import get_logger


def build_oare_pairs(
    sentences_path: str | Path,
    published_texts_path: str | Path,
    train_ids: set[str],
    min_translation_len: int = 20,
    min_translit_len: int = 20,
    max_translit_len: int = 3000,
) -> pd.DataFrame:
    """Cross-reference Sentences_Oare with published_texts to build new pairs.

    Groups sentence-level translations by text_uuid, concatenates them,
    and pairs with the full tablet transliteration from published_texts.
    Excludes tablets already in the training set.
    """
    so = pd.read_csv(sentences_path)
    pt = pd.read_csv(published_texts_path)

    # Keep only texts NOT in train
    new_ids = set(so["text_uuid"].dropna()) - train_ids
    so_new = so[so["text_uuid"].isin(new_ids) & so["translation"].notna()]

    # Sort by position in text for correct sentence ordering
    so_new = so_new.sort_values(["text_uuid", "sentence_obj_in_text"])

    # Concatenate sentence translations per tablet
    grouped = (
        so_new.groupby("text_uuid")
        .agg(translation=("translation", lambda x: " ".join(x.dropna().astype(str))))
        .reset_index()
    )

    # Join with published_texts for transliterations
    joined = grouped.merge(
        pt[["oare_id", "transliteration"]],
        left_on="text_uuid",
        right_on="oare_id",
        how="inner",
    )

    # Quality filters
    joined = joined[
        (joined["translation"].str.len() >= min_translation_len)
        & (joined["transliteration"].str.len() >= min_translit_len)
        & (joined["transliteration"].str.len() <= max_translit_len)
    ]

    return joined[["oare_id", "transliteration", "translation"]].reset_index(drop=True)


def build_dictionary_pairs(
    dict_path: str | Path,
    min_def_len: int = 5,
) -> pd.DataFrame:
    """Build word→definition pairs from eBL Dictionary for lexical augmentation."""
    ebl = pd.read_csv(dict_path)
    ebl = ebl[ebl["definition"].notna() & (ebl["definition"].str.len() >= min_def_len)]

    # Clean word entries — strip morphological markers
    ebl = ebl.copy()
    ebl["word_clean"] = ebl["word"].str.replace(r"\s+[IVX]+$", "", regex=True).str.strip()

    pairs = pd.DataFrame({
        "oare_id": [f"ebl_{i}" for i in range(len(ebl))],
        "transliteration": ebl["word_clean"].values,
        "translation": ebl["definition"].values,
    })
    return pairs.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--include-dictionary", action="store_true",
                        help="Include eBL Dictionary word→definition pairs")
    args = parser.parse_args()

    logger = get_logger()
    data_cfg = load_yaml(args.data_config)

    # Load existing train IDs to exclude
    train = pd.read_csv(data_cfg["paths"]["train_csv"])
    train_ids = set(train["oare_id"].astype(str))
    logger.info("Existing train tablets: %d", len(train_ids))

    all_pairs = []

    # Sentences_Oare cross-reference
    oare_pairs = build_oare_pairs(
        sentences_path=data_cfg["paths"]["sentences_csv"],
        published_texts_path=data_cfg["paths"]["published_texts_csv"],
        train_ids=train_ids,
    )
    logger.info("OARE expansion pairs: %d", len(oare_pairs))
    all_pairs.append(oare_pairs)

    # Dictionary pairs (optional)
    if args.include_dictionary:
        dict_pairs = build_dictionary_pairs(data_cfg["paths"]["dictionary_csv"])
        logger.info("Dictionary pairs: %d", len(dict_pairs))
        all_pairs.append(dict_pairs)

    expanded = pd.concat(all_pairs, ignore_index=True)
    logger.info("Total augmented pairs: %d", len(expanded))

    # Write output
    out_path = Path(data_cfg["paths"]["augmented_csv"])
    ensure_dir(out_path.parent)
    expanded.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
