"""Lexicon-aware BIO label generation for auxiliary heads.

Loads the OA Lexicon (type = PN / GN / word) and produces byte-level BIO tags
for name spans, number spans, and damage markers in Akkadian transliteration.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

# BIO tag indices
O, B, I = 0, 1, 2

# ── Lexicon loading ──

_PN_FORMS: Set[str] = set()
_GN_FORMS: Set[str] = set()
_LEXICON_LOADED = False


def load_lexicon(lexicon_path: str | Path) -> None:
    """Load the OA_Lexicon_eBL.csv to populate PN and GN form sets."""
    global _PN_FORMS, _GN_FORMS, _LEXICON_LOADED
    if _LEXICON_LOADED:
        return
    path = Path(lexicon_path)
    if not path.exists():
        return
    pn = set()
    gn = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            form = (row.get("form") or "").strip().lower()
            if not form:
                continue
            typ = (row.get("type") or "").strip()
            if typ == "PN":
                pn.add(form)
            elif typ == "GN":
                gn.add(form)
    _PN_FORMS = pn
    _GN_FORMS = gn
    _LEXICON_LOADED = True


def _is_name_word(word: str) -> bool:
    """Check if a transliteration word is a proper name using the lexicon + heuristics."""
    # Strip determinatives like {d}, {m}, {f}, {ki}
    clean = re.sub(r"\{[^}]*\}", "", word).strip()
    if not clean:
        return False
    low = clean.lower()
    # Check lexicon
    if low in _PN_FORMS or low in _GN_FORMS:
        return True
    # Heuristic: first letter capitalized but not ALL-CAPS (Sumerograms)
    if clean[0].isupper() and not clean.isupper():
        return True
    # Preceded by determinatives {d}, {m}, {f} → usually personal name
    if re.match(r"\{[dmf]\}", word):
        return True
    return False


# ── Number detection ──
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Damage markers in transliteration
_DAMAGE_CHARS = set("[]?")
_DAMAGE_TOKENS = {"<gap>", "<big_gap>"}


def generate_bio_labels(source_text: str, n_bytes: int) -> Dict[str, List[int]]:
    """Generate byte-level BIO labels for name, number, and damage spans.

    Args:
        source_text: The packed source text (may include <raw>/<norm> tags).
        n_bytes: Total length including BOS/EOS tokens from ByteSourceEncoder.

    Returns:
        Dict with keys 'name_labels', 'number_labels', 'damage_labels',
        each a list of ints of length n_bytes with values O=0, B=1, I=2.
    """
    name_labels = [O] * n_bytes
    number_labels = [O] * n_bytes
    damage_labels = [O] * n_bytes

    # Work on the raw bytes (offset +1 for BOS, +3 for byte encoding offset)
    raw_bytes = source_text.encode("utf-8", errors="replace")

    # ── Name labels via word-level analysis ──
    # We iterate through words and map character positions → byte positions
    char_to_byte = []
    byte_pos = 0
    for ch in source_text:
        ch_bytes = ch.encode("utf-8", errors="replace")
        for _ in ch_bytes:
            pass  # count bytes
        char_to_byte.append((byte_pos, byte_pos + len(ch_bytes)))
        byte_pos += len(ch_bytes)

    # Word boundaries
    words = _split_words_with_positions(source_text)
    for word, char_start, char_end in words:
        if _is_name_word(word):
            _mark_span_bio(name_labels, char_to_byte, char_start, char_end)

    # ── Number labels ──
    for m in _NUMBER_RE.finditer(source_text):
        _mark_span_bio(number_labels, char_to_byte, m.start(), m.end())

    # ── Damage labels ──
    # Mark damage characters and special tokens
    for i, ch in enumerate(source_text):
        if ch in _DAMAGE_CHARS:
            _mark_char_bio(damage_labels, char_to_byte, i)

    # Mark <gap> and <big_gap> tokens
    for token in _DAMAGE_TOKENS:
        start = 0
        while True:
            idx = source_text.find(token, start)
            if idx == -1:
                break
            _mark_span_bio(damage_labels, char_to_byte, idx, idx + len(token))
            start = idx + len(token)

    return {
        "name_labels": name_labels,
        "number_labels": number_labels,
        "damage_labels": damage_labels,
    }


def _split_words_with_positions(text: str) -> List[Tuple[str, int, int]]:
    """Split text into words with their character positions."""
    results = []
    for m in re.finditer(r"\S+", text):
        results.append((m.group(), m.start(), m.end()))
    return results


def _mark_span_bio(
    labels: List[int],
    char_to_byte: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> None:
    """Mark a character span as B-I-I... in byte-level labels (offset +1 for BOS)."""
    if char_start >= len(char_to_byte) or char_end > len(char_to_byte):
        return
    first = True
    for ci in range(char_start, min(char_end, len(char_to_byte))):
        byte_start, byte_end = char_to_byte[ci]
        for bi in range(byte_start, byte_end):
            idx = bi + 1  # +1 for BOS token
            if idx < len(labels):
                labels[idx] = B if first else I
                first = False


def _mark_char_bio(
    labels: List[int],
    char_to_byte: List[Tuple[int, int]],
    char_idx: int,
) -> None:
    """Mark a single character position as B in byte-level labels."""
    if char_idx >= len(char_to_byte):
        return
    byte_start, byte_end = char_to_byte[char_idx]
    for bi in range(byte_start, byte_end):
        idx = bi + 1  # +1 for BOS
        if idx < len(labels):
            labels[idx] = B if bi == byte_start else I
