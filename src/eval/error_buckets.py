from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import pandas as pd

from src.eval.metrics import extract_name_spans, extract_numbers


@dataclass
class ErrorRow:
    text_id: str
    bucket: str
    source: str
    prediction: str
    reference: str


def build_error_buckets(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[ErrorRow] = []
    for _, row in df.iterrows():
        pred = str(row["prediction"])
        ref = str(row["reference"])
        src = str(row["source"])
        text_id = str(row["text_id"])

        if extract_numbers(pred) != extract_numbers(ref):
            rows.append(ErrorRow(text_id, "wrong_numbers", src, pred, ref))

        if not extract_name_spans(pred).issuperset(extract_name_spans(ref)):
            rows.append(ErrorRow(text_id, "dropped_names", src, pred, ref))

        if len(pred.split()) > 1.5 * max(1, len(ref.split())):
            rows.append(ErrorRow(text_id, "overlong_output", src, pred, ref))

        if any(tok in src for tok in ["[", "]", " x ", "?"]) and len(pred.split()) > len(ref.split()) + 5:
            rows.append(ErrorRow(text_id, "damage_hallucination", src, pred, ref))

    return pd.DataFrame([asdict(x) for x in rows])
