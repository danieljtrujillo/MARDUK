from __future__ import annotations

from typing import Iterable


def build_metadata_prefix(row: dict, metadata_columns: Iterable[str]) -> str:
    parts: list[str] = []
    for col in metadata_columns:
        value = row.get(col)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        parts.append(f"<{col}={value}>")
    return " ".join(parts).strip()


def pack_source(
    raw_text: str,
    normalized_text: str | None = None,
    metadata_prefix: str | None = None,
    include_raw_view: bool = True,
    include_normalized_view: bool = False,
    wrap_views: bool = True,
) -> str:
    parts: list[str] = []

    if metadata_prefix:
        parts.append(metadata_prefix)

    if include_raw_view:
        parts.append(f"<raw> {raw_text} </raw>" if wrap_views else raw_text)

    if include_normalized_view and normalized_text:
        parts.append(f"<norm> {normalized_text} </norm>" if wrap_views else normalized_text)

    return " ".join(part for part in parts if part).strip()
