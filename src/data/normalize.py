from __future__ import annotations

import re
import unicodedata


DASH_TRANSLATION = str.maketrans({
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
    "\u2212": "-",   # minus sign
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
})

# ── Competition-specific character mappings ──
# Per competition Dataset Instructions: Ḫ ḫ → H h  (only one type of H in Akkadian)
AKKADIAN_CHAR_MAP = str.maketrans({
    "\u1e2a": "H",   # Ḫ
    "\u1e2b": "h",   # ḫ
})

# Subscript digits → ASCII digits  (e.g. il₅ → il5)
SUBSCRIPT_MAP = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u2093",
    "0123456789x",
)

# Superscript digits → ASCII digits
SUPERSCRIPT_MAP = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
    "0123456789",
)

# Accented vowel → numbered form (canonical transliteration)
VOWEL_MAP = {
    "\u00e1": "a2", "\u00e0": "a3",   # á → a2, à → a3
    "\u00e9": "e2", "\u00e8": "e3",   # é → e2, è → e3
    "\u00ed": "i2", "\u00ec": "i3",   # í → i2, ì → i3
    "\u00fa": "u2", "\u00f9": "u3",   # ú → u2, ù → u3
}


def normalize_akkadian_chars(text: str) -> str:
    """Apply competition-specific character normalizations for Akkadian transliteration."""
    text = text.translate(AKKADIAN_CHAR_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    for src, dst in VOWEL_MAP.items():
        text = text.replace(src, dst)
    return text


def normalize_scribal_notations(text: str) -> str:
    """Remove modern scribal notations per competition formatting suggestions."""
    # Remove certain-reading marks !
    text = text.replace("!", "")
    # Remove half-brackets (partially broken signs)
    text = re.sub(r"[\u02f9\u2e22\u2e23\u0304]", "", text)   # ˹ ˺
    text = re.sub(r"[\u2e22\u2e23]", "", text)
    text = text.replace("\u2e22", "").replace("\u2e23", "")
    text = text.replace("\u0338", "").replace("\u0339", "")
    # ⸢ ⸣ variants
    text = text.replace("\u2e22", "").replace("\u2e23", "")
    # Remove erroneous sign brackets: <<text>> → text  (before single < >)
    text = re.sub(r"<<([^>]*)>>", r"\1", text)
    # Remove scribal insertion brackets but keep content: <text> → text
    # Exclude our special tags <gap>, <big_gap>, <raw>, <norm>, etc.
    text = re.sub(r"<(?!/?(?:gap|big_gap|raw|norm|gloss)\b)([^>]*)>", r"\1", text)
    # Normalize breaks: [x] → <gap>,  [... ...] or [...] or … → <big_gap>
    text = re.sub(r"\[x\]", "<gap>", text)
    text = re.sub(r"\[\.\.\.\s*\.\.\.\]", "<big_gap>", text)
    text = re.sub(r"\[\.\.\.\]", "<big_gap>", text)
    text = re.sub(r"\u2026+", "<big_gap>", text)  # …
    # Remove square brackets around readable text: [KÙ.BABBAR] → KÙ.BABBAR
    text = re.sub(r"\[([^\[\]]*?)\]", r"\1", text)
    # Remove line dividers / and word dividers : (standalone)
    text = re.sub(r"\s/\s", " ", text)
    text = re.sub(r"\s:\s", " ", text)
    # Remove question marks used as uncertain readings
    text = re.sub(r"\?(?=\s|$)", "", text)
    return text


def normalize_line_numbers(text: str) -> str:
    """Strip line number prefixes like '1.', '5'.', '10''. from start of lines."""
    text = re.sub(r"(?m)^\s*\d+['\u2019]*[\.\)]\s*", "", text)
    return text


def normalize_text(
    text: str,
    lowercase: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode_punctuation: bool = True,
    space_repeated_separators: bool = True,
    preserve_damage_markers: bool = True,
    apply_akkadian_normalization: bool = True,
) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = unicodedata.normalize("NFKC", text)

    if apply_akkadian_normalization:
        text = normalize_akkadian_chars(text)
        text = normalize_scribal_notations(text)
        text = normalize_line_numbers(text)

    if normalize_unicode_punctuation:
        text = text.translate(DASH_TRANSLATION)

    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    if space_repeated_separators:
        text = re.sub(r"([/|:;,\-])\1+", r"\1 \1", text)

    if not preserve_damage_markers:
        text = re.sub(r"[\[\]\u2e22\u2e23\u27e6\u27e7?xX]+", " ", text)

    if lowercase:
        text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()
    return text
