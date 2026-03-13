from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class ColumnConfig:
    source: str
    target: str | None
    id: str | None = None
    metadata: list[str] | None = None


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, columns: ColumnConfig) -> None:
    required = [columns.source]
    if columns.target:
        required.append(columns.target)
    if columns.id:
        required.append(columns.id)
    if columns.metadata:
        required.extend(columns.metadata)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def materialize_examples(df: pd.DataFrame, columns: ColumnConfig) -> pd.DataFrame:
    validate_columns(df, columns)
    out = pd.DataFrame()
    out["source_text"] = df[columns.source].fillna("").astype(str)
    out["target_text"] = df[columns.target].fillna("").astype(str) if columns.target else ""
    out["text_id"] = (
        df[columns.id].astype(str)
        if columns.id and columns.id in df.columns
        else pd.Series([str(i) for i in range(len(df))], index=df.index)
    )
    metadata_cols = columns.metadata or []
    for col in metadata_cols:
        out[col] = df[col]
    return out
