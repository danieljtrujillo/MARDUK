from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.build_dual_view import build_metadata_prefix, pack_source
from src.data.load_kaggle import ColumnConfig, materialize_examples, read_csv
from src.data.normalize import normalize_text, clean_translation
from src.data.splitters import FoldConfig, add_kfold_column, add_group_kfold_column
from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--view-config", required=True)
    return parser.parse_args()


def _load_augmented(path: str | Path) -> pd.DataFrame | None:
    """Load augmented_pairs.csv if it exists. Expected columns: transliteration, translation."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Rename to match expected schema if needed
    rename_map = {}
    if "akkadian" in df.columns and "transliteration" not in df.columns:
        rename_map["akkadian"] = "transliteration"
    if "english" in df.columns and "translation" not in df.columns:
        rename_map["english"] = "translation"
    if rename_map:
        df = df.rename(columns=rename_map)
    # Ensure required columns
    if "transliteration" not in df.columns or "translation" not in df.columns:
        return None
    # Add oare_id if missing
    if "oare_id" not in df.columns:
        df["oare_id"] = [f"aug_{i}" for i in range(len(df))]
    return df


def _apply_normalization(frame: pd.DataFrame, norm_cfg: dict) -> pd.DataFrame:
    """Apply text normalization to source_text column, creating normalized_text."""
    frame["normalized_text"] = frame["source_text"].map(
        lambda x: normalize_text(
            x,
            lowercase=norm_cfg["lowercase"],
            normalize_whitespace=norm_cfg["normalize_whitespace"],
            normalize_unicode_punctuation=norm_cfg["normalize_unicode_punctuation"],
            space_repeated_separators=norm_cfg["space_repeated_separators"],
            preserve_damage_markers=norm_cfg["preserve_damage_markers"],
            apply_akkadian_normalization=True,
        )
    )
    return frame


def _apply_packing(frame: pd.DataFrame, data_cfg: dict, view_cfg: dict) -> pd.DataFrame:
    """Build metadata prefix and pack source text."""
    metadata_cols = data_cfg["columns"].get("metadata", [])
    frame["metadata_prefix"] = frame.apply(
        lambda row: build_metadata_prefix(row.to_dict(), metadata_cols),
        axis=1,
    )
    packing = view_cfg["packing"]
    frame["packed_source"] = frame.apply(
        lambda row: pack_source(
            raw_text=row["source_text"],
            normalized_text=row["normalized_text"],
            metadata_prefix=row["metadata_prefix"] if packing["include_metadata"] else None,
            include_raw_view=packing["include_raw_view"],
            include_normalized_view=packing["include_normalized_view"],
            wrap_views=packing["wrap_views"],
        ),
        axis=1,
    )
    return frame


def main() -> None:
    args = parse_args()
    logger = get_logger()

    data_cfg = load_yaml(args.data_config)
    view_cfg = load_yaml(args.view_config)
    processed_dir = ensure_dir(data_cfg["paths"]["processed_dir"])

    # ── Load train data ──
    train_df = read_csv(data_cfg["paths"]["train_csv"])

    # ── Fix truncated transliterations using published_texts ──
    pt_path = data_cfg["paths"].get("published_texts_csv")
    if pt_path and Path(pt_path).exists():
        pt_df = pd.read_csv(pt_path)
        id_col = data_cfg["columns"].get("id", "oare_id")
        src_col = data_cfg["columns"]["source"]
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
                logger.info("Fixed %d truncated transliterations from published_texts", n_fixed)

    columns = ColumnConfig(
        source=data_cfg["columns"]["source"],
        target=data_cfg["columns"].get("target"),
        id=data_cfg["columns"].get("id"),
        metadata=data_cfg["columns"].get("metadata", []),
    )
    train = materialize_examples(train_df, columns)
    train["data_source"] = "train"
    # Apply v3 competition cleaning to translation targets
    train["target_text"] = train["target_text"].map(clean_translation)
    logger.info("Loaded %d training examples from train.csv", len(train))

    # ── Load augmented data if available ──
    aug_path = data_cfg["paths"].get("augmented_csv")
    if aug_path:
        aug_df = _load_augmented(aug_path)
        if aug_df is not None:
            aug = materialize_examples(
                aug_df,
                ColumnConfig(
                    source=data_cfg["columns"]["source"],
                    target=data_cfg["columns"].get("target"),
                    id=data_cfg["columns"].get("id"),
                    metadata=[],  # augmented data won't have metadata columns
                ),
            )
            aug["data_source"] = "augmented"
            aug["target_text"] = aug["target_text"].map(clean_translation)
            train = pd.concat([train, aug], ignore_index=True)
            logger.info("Added %d augmented examples (total: %d)", len(aug), len(train))

    # ── Load test data ──
    test_df = read_csv(data_cfg["paths"]["test_csv"])
    # Test has different column names: id, text_id, transliteration
    test_id_col = data_cfg["columns"].get("test_id", "id")
    test_source_col = data_cfg["columns"].get("test_source", data_cfg["columns"]["source"])
    test_text_id_col = data_cfg["columns"].get("test_text_id", "text_id")
    test = materialize_examples(
        test_df,
        ColumnConfig(
            source=test_source_col,
            target=None,
            id=test_id_col,
            metadata=[],
        ),
    )
    # Preserve text_id for grouping during inference
    if test_text_id_col in test_df.columns:
        test["tablet_text_id"] = test_df[test_text_id_col].astype(str).values
    # Preserve line_start, line_end for ordering
    for col in ["line_start", "line_end"]:
        if col in test_df.columns:
            test[col] = test_df[col].values

    # ── Normalize & pack ──
    norm_cfg = data_cfg["normalization"]
    for frame in (train, test):
        _apply_normalization(frame, norm_cfg)
        _apply_packing(frame, data_cfg, view_cfg)

    # ── Drop rows with missing target_text (unusable for training) ──
    n_before = len(train)
    train = train[train["target_text"].str.strip().astype(bool)]
    n_dropped = n_before - len(train)
    if n_dropped:
        logger.info("Dropped %d rows with missing/empty target_text", n_dropped)

    # ── Add fold column (GroupKFold by text_id to prevent tablet leakage) ──
    split_cfg = data_cfg["splits"]
    folds = FoldConfig(
        n_splits=split_cfg["n_splits"],
        random_state=split_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
    )
    train = add_group_kfold_column(train, folds, group_col="text_id")

    # ── Write outputs ──
    train.to_csv(processed_dir / "train_prepared.csv", index=False)
    test.to_csv(processed_dir / "test_prepared.csv", index=False)

    logger.info("Wrote %s (%d rows)", processed_dir / "train_prepared.csv", len(train))
    logger.info("Wrote %s (%d rows)", processed_dir / "test_prepared.csv", len(test))

    # Print fold distribution
    if "fold" in train.columns:
        for fold_id in sorted(train["fold"].unique()):
            fold_size = len(train[train["fold"] == fold_id])
            logger.info("  Fold %d: %d examples", fold_id, fold_size)


if __name__ == "__main__":
    main()
