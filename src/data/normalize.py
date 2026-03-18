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

# NOTE: We intentionally preserve diacritics (á à é è í ì ú ù) in transliterations.
# The competition test data uses diacritics, NOT numbered forms (a2/a3).
# Converting them would cause format mismatch and severe score loss.


def normalize_akkadian_chars(text: str) -> str:
    """Apply competition-specific character normalizations for Akkadian transliteration."""
    text = text.translate(AKKADIAN_CHAR_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    # Diacritics (á à é è í ì ú ù) are preserved — test data uses them
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
    # Exclude our special tags <gap>, <raw>, <norm>, etc.
    text = re.sub(r"<(?!/?(?:gap|raw|norm|gloss)\b)([^>]*)>", r"\1", text)
    # Normalize ALL breaks to single <gap> (no <big_gap> per v3 rules)
    text = re.sub(r"\[x\]", "<gap>", text)
    text = re.sub(r"\[\.\.\.\s*\.\.\.\]", "<gap>", text)
    text = re.sub(r"\[\.\.\.\]", "<gap>", text)
    text = re.sub(r"\u2026+", "<gap>", text)  # …
    text = re.sub(r"\(break\)", "<gap>", text)
    text = re.sub(r"\(large break\)", "<gap>", text)
    text = re.sub(r"\(\d+ broken lines\)", "<gap>", text)
    # Replace <big_gap> → <gap>
    text = text.replace("<big_gap>", "<gap>")
    # Deduplicate adjacent <gap> tokens (with optional dashes/spaces between)
    text = re.sub(r"(<gap>)(?:[\s\-]*<gap>)+", r"\1", text)
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


# ── Decimal-to-fraction mapping (per competition v3 recommendations) ──
DECIMAL_TO_FRACTION = {
    "0.5": "½", "0.25": "¼", "0.75": "¾",
    "0.3333": "⅓", "0.33333": "⅓", "0.333": "⅓",
    "0.6666": "⅔", "0.66666": "⅔", "0.667": "⅔",
    "0.1666": "⅙", "0.16666": "⅙", "0.167": "⅙",
    "0.8333": "⅚", "0.83333": "⅚", "0.833": "⅚",
    "0.625": "⅝",
}

# Roman numeral to integer for months
_ROMAN_TO_INT = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6",
    "VII": "7", "VIII": "8", "IX": "9", "X": "10", "XI": "11", "XII": "12",
}


def clean_translation(text: str) -> str:
    """Apply competition v3 recommended cleaning to translation text.

    This handles: fem./sing./pl. removal, PN→<gap>, decimal→fraction,
    month Roman→integer, -textiles/-gold/-tax expansion, straight quotes,
    (?) removal, gap deduplication.
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)

    # Straight quotes (curly → straight)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove scribal annotations: fem., sing., pl., plural
    text = re.sub(r"\bfem\.\s*", "", text)
    text = re.sub(r"\bsing\.\s*", "", text)
    text = re.sub(r"\bpl\.\s*", "", text)
    text = re.sub(r"\bplural\b\s*", "", text)

    # Remove (?) markers
    text = re.sub(r"\(\?\)", "", text)

    # PN → <gap>
    text = re.sub(r"\bPN\b", "<gap>", text)

    # -textiles → kutānum textiles  (only when preceded by space, i.e. standalone)
    text = re.sub(r"(?<=\s)-textiles\b", "kutānum textiles", text)
    # -gold → pašallum gold
    text = re.sub(r"(?<=\s)-gold\b", "pašallum gold", text)
    # -tax → šadduātum tax
    text = re.sub(r"(?<=\s)-tax\b", "šadduātum tax", text)

    # Decimal fractions → unicode fractions (translation only)
    # Sort by length descending to match longer decimals first
    for dec, frac in sorted(DECIMAL_TO_FRACTION.items(), key=lambda x: -len(x[0])):
        text = text.replace(dec, frac)

    # Long floats with many decimal places → fraction
    text = re.sub(r"1\.3333\d*", "1⅓", text)
    text = re.sub(r"1\.6666\d*", "1⅔", text)
    text = re.sub(r"2\.6666\d*", "2⅔", text)
    text = re.sub(r"(\d+)\.3333\d*", lambda m: m.group(1) + "⅓", text)
    text = re.sub(r"(\d+)\.6666\d*", lambda m: m.group(1) + "⅔", text)
    text = re.sub(r"(\d+)\.8333\d*", lambda m: m.group(1) + "⅚", text)
    text = re.sub(r"(\d+)\.1666\d*", lambda m: m.group(1) + "⅙", text)

    # Month Roman numerals → integers (e.g. "month V" → "month 5", "Month XII" → "Month 12")
    def _replace_month_roman(m):
        prefix = m.group(1)  # "month" or "Month"
        roman = m.group(2)
        return prefix + " " + _ROMAN_TO_INT.get(roman, roman)
    text = re.sub(r"\b((?:M|m)onth)\s+(XII|XI|IX|VIII|VII|VI|IV|III|II|X|V|I)\b", _replace_month_roman, text)

    # Replace <big_gap> → <gap> in translations too
    text = text.replace("<big_gap>", "<gap>")

    # Deduplicate adjacent <gap>
    text = re.sub(r"(<gap>)(?:[\s\-]*<gap>)+", r"\1", text)

    # Ḫ/ḫ → H/h  (not present in test translations per host)
    text = text.replace("\u1e2a", "H").replace("\u1e2b", "h")

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
