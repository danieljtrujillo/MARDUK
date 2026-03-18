# MARDUK – Akkadian → English Neural Translation
# Competition: Deep Past Initiative – Machine Translation (Kaggle)
#
# This notebook loads a fine-tuned ByT5-base model (v2 clean, val=43.26)
# and generates translations for the test set.
#
# Model: google/byt5-base retrained on clean data (no augmented), TGT_MAX=256
# Data source: theskateborg/marduk-model-download (Kaggle kernel output)
# Architecture: Byte-level Seq2Seq with dual-view (raw + normalized) input packing

# %% [markdown]
# # MARDUK – Akkadian to English Translation

# %%
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# %% [markdown]
# ## 1. Configuration

# %%
# Model path – expects model files in Kaggle dataset
MODEL_PATHS = [
    Path("/kaggle/input/notebooks/theskateborg/marduk-model-download/model"),  # kernel output
    Path("/kaggle/input/marduk-model-download/model"),     # alt kernel output
    Path("/kaggle/input/marduk-byt5-akkadian2english"),     # from dataset
    Path("/kaggle/input/marduk-model-download"),            # flat layout
]
MODEL_PATH = None
for p in MODEL_PATHS:
    if p.exists() and (p / "config.json").exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    # List available inputs for debugging
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for item in sorted(input_dir.rglob("*")):
            print(item)
    raise FileNotFoundError("Model not found in any expected path")

print(f"Model path: {MODEL_PATH}")

# Competition data - try both path formats
for _test_path in [
    Path("/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"),
    Path("/kaggle/input/deep-past-initiative-machine-translation/test.csv"),
]:
    if _test_path.exists():
        TEST_CSV = _test_path
        break
else:
    TEST_CSV = Path("/kaggle/input/deep-past-initiative-machine-translation/test.csv")
OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# Generation parameters — match training eval config that scored val=43.26
SRC_MAX = 1024
TGT_MAX = 512  # bumped from 256 — training targets often exceed 256 bytes
NUM_BEAMS = 5
BATCH_SIZE = 4

# MBR self-ensemble parameters
MBR_SAMPLES = 12       # number of diverse candidates per input (5 beam + 7 sampled)
MBR_TEMPERATURE = 0.8  # sampling temperature for diversity
MBR_TOP_P = 0.92       # nucleus sampling threshold
MBR_SAMPLE_BATCH = 4   # generate samples in small batches to avoid OOM

# %% [markdown]
# ## 2. Preprocessing – Akkadian Normalization & Dual-View Packing

# %%
# ── Character maps ──
DASH_TRANSLATION = str.maketrans({
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
})
AKKADIAN_CHAR_MAP = str.maketrans({"\u1e2a": "H", "\u1e2b": "h"})
SUBSCRIPT_MAP = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u2093",
    "0123456789x",
)
SUPERSCRIPT_MAP = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
    "0123456789",
)
# NOTE: Diacritics (á à é è í ì ú ù) are preserved — test data uses them

def normalize_akkadian_chars(text: str) -> str:
    text = text.translate(AKKADIAN_CHAR_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    return text

def normalize_scribal_notations(text: str) -> str:
    text = text.replace("!", "")
    text = re.sub(r"[\u02f9\u2e22\u2e23\u0304]", "", text)
    text = re.sub(r"[\u2e22\u2e23]", "", text)
    text = text.replace("\u2e22", "").replace("\u2e23", "")
    text = text.replace("\u0338", "").replace("\u0339", "")
    text = re.sub(r"<<([^>]*)>>", r"\1", text)
    text = re.sub(r"<(?!/?(?:gap|raw|norm|gloss)\b)([^>]*)>", r"\1", text)
    text = re.sub(r"\[x\]", "<gap>", text)
    text = re.sub(r"\[\.\.\.\s*\.\.\.\]", "<gap>", text)
    text = re.sub(r"\[\.\.\.\]", "<gap>", text)
    text = re.sub(r"\u2026+", "<gap>", text)
    text = re.sub(r"\(break\)", "<gap>", text)
    text = re.sub(r"\(large break\)", "<gap>", text)
    text = re.sub(r"\(\d+ broken lines\)", "<gap>", text)
    text = text.replace("<big_gap>", "<gap>")
    text = re.sub(r"(<gap>)(?:[\s\-]*<gap>)+", r"\1", text)
    text = re.sub(r"\[([^\[\]]*?)\]", r"\1", text)
    text = re.sub(r"\s/\s", " ", text)
    text = re.sub(r"\s:\s", " ", text)
    text = re.sub(r"\?(?=\s|$)", "", text)
    return text

def normalize_line_numbers(text: str) -> str:
    return re.sub(r"(?m)^\s*\d+['\u2019]*[\.\)]\s*", "", text)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = normalize_akkadian_chars(text)
    text = normalize_scribal_notations(text)
    text = normalize_line_numbers(text)
    text = text.translate(DASH_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([/|:;,\-])\1+", r"\1 \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pack_source(raw_text: str, normalized_text: str) -> str:
    """Create dual-view input: <raw> ... </raw> <norm> ... </norm>"""
    return f"<raw> {raw_text} </raw> <norm> {normalized_text} </norm>"

def preprocess_row(row: dict) -> str:
    source = str(row.get("transliteration", ""))
    normalized = normalize_text(source)
    return pack_source(raw_text=source, normalized_text=normalized)

# %% [markdown]
# ## 3. Load Model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH)).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params / 1e6:.1f}M parameters")

# %% [markdown]
# ## 4. Load & Preprocess Test Data

# %%
test_df = pd.read_csv(TEST_CSV)
print(f"Test set: {len(test_df)} examples")
print(f"Columns: {test_df.columns.tolist()}")

packed_sources = [preprocess_row(row.to_dict()) for _, row in test_df.iterrows()]

# Sort by text_id for sibling context during generation
if "text_id" in test_df.columns:
    test_df["_packed"] = packed_sources
    sort_cols = ["text_id"]
    if "line_start" in test_df.columns:
        sort_cols.append("line_start")
    test_df = test_df.sort_values(sort_cols)
    packed_sources = test_df["_packed"].tolist()
    test_df = test_df.drop(columns=["_packed"])

print(f"\nSample input:\n{packed_sources[0][:200]}...")

# %% [markdown]
# ## 5. Generate Translations (MBR Self-Ensemble)

# %%
# ── Inline chrF++ scorer (no external deps) ──
from collections import Counter

def _extract_char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams from text."""
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))

def _extract_word_ngrams(text: str, n: int) -> Counter:
    """Extract word n-grams from text."""
    words = text.split()
    return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def chrf_score(hypothesis: str, reference: str,
               char_order: int = 6, word_order: int = 2,
               beta: float = 2.0) -> float:
    """Compute chrF++ between two strings.
    
    chrF++ uses character n-grams (1..char_order) + word n-grams (1..word_order).
    F_beta with beta=2 weights recall twice as much as precision.
    """
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0

    total_precision_num = 0.0
    total_precision_den = 0.0
    total_recall_num = 0.0
    total_recall_den = 0.0
    n_total = 0

    # Character n-grams
    for n in range(1, char_order + 1):
        hyp_ngrams = _extract_char_ngrams(hypothesis, n)
        ref_ngrams = _extract_char_ngrams(reference, n)
        if not hyp_ngrams and not ref_ngrams:
            continue
        common = sum((hyp_ngrams & ref_ngrams).values())
        total_precision_num += common
        total_precision_den += sum(hyp_ngrams.values())
        total_recall_num += common
        total_recall_den += sum(ref_ngrams.values())
        n_total += 1

    # Word n-grams (the ++ in chrF++)
    for n in range(1, word_order + 1):
        hyp_ngrams = _extract_word_ngrams(hypothesis, n)
        ref_ngrams = _extract_word_ngrams(reference, n)
        if not hyp_ngrams and not ref_ngrams:
            continue
        common = sum((hyp_ngrams & ref_ngrams).values())
        total_precision_num += common
        total_precision_den += sum(hyp_ngrams.values())
        total_recall_num += common
        total_recall_den += sum(ref_ngrams.values())
        n_total += 1

    if n_total == 0:
        return 0.0

    precision = total_precision_num / total_precision_den if total_precision_den > 0 else 0.0
    recall = total_recall_num / total_recall_den if total_recall_den > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return score * 100  # Scale to 0-100 like sacrebleu


def mbr_select(candidates):
    """Select the MBR-optimal candidate: the one with highest average chrF++
    against all other candidates (consensus translation).

    Returns (best_candidate, best_score).
    """
    if len(candidates) <= 1:
        return candidates[0] if candidates else "", 0.0

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        c_stripped = c.strip()
        if c_stripped not in seen:
            seen.add(c_stripped)
            unique.append(c_stripped)

    if len(unique) == 1:
        return unique[0], 100.0

    n = len(unique)
    # Compute pairwise chrF++ matrix (symmetric)
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = chrf_score(unique[i], unique[j])
            scores[i][j] = s
            scores[j][i] = s

    # Average score for each candidate (against all others)
    avg_scores = [sum(scores[i]) / (n - 1) for i in range(n)]
    best_idx = max(range(n), key=lambda i: avg_scores[i])
    return unique[best_idx], avg_scores[best_idx]


@torch.no_grad()
def generate_candidates(packed_source):
    """Generate diverse translation candidates for one input using
    beam search + sampling for MBR selection."""
    inputs = tokenizer(
        [packed_source],
        max_length=SRC_MAX,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    candidates = []

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        # 1) Beam search — get top beam hypotheses
        beam_out = model.generate(
            **inputs,
            max_new_tokens=TGT_MAX,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            early_stopping=True,
        )
        beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)
        candidates.extend(beam_texts)
        del beam_out

        # 2) Diverse sampling — get stochastic candidates in small batches
        n_samples = MBR_SAMPLES - NUM_BEAMS
        remaining = n_samples
        while remaining > 0:
            batch_n = min(remaining, MBR_SAMPLE_BATCH)
            sample_out = model.generate(
                **inputs,
                max_new_tokens=TGT_MAX,
                do_sample=True,
                temperature=MBR_TEMPERATURE,
                top_p=MBR_TOP_P,
                num_return_sequences=batch_n,
            )
            sample_texts = tokenizer.batch_decode(sample_out, skip_special_tokens=True)
            candidates.extend(sample_texts)
            del sample_out
            remaining -= batch_n

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


# Generate + MBR-select for each test example
all_predictions = []
for idx, packed in enumerate(tqdm(packed_sources, desc="MBR Translating")):
    try:
        candidates = generate_candidates(packed)
        best, score = mbr_select(candidates)
        print(f"  Example {idx}: {len(candidates)} cands, "
              f"{len(set(c.strip() for c in candidates))} unique, "
              f"MBR={score:.1f}")
    except Exception as e:
        # Fallback to simple beam search if MBR fails (e.g. OOM)
        print(f"  Example {idx}: MBR failed ({e}), falling back to beam")
        torch.cuda.empty_cache()
        inputs = tokenizer(
            [packed], max_length=SRC_MAX, truncation=True,
            padding=True, return_tensors="pt"
        ).to(device)
        out = model.generate(**inputs, max_new_tokens=TGT_MAX,
                             num_beams=NUM_BEAMS, early_stopping=True)
        best = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    all_predictions.append(best)

print(f"\nGenerated {len(all_predictions)} translations via MBR self-ensemble")

# %% [markdown]
# ## 6. Post-Process & Create Submission

# %%
# ── Gap normalization regex (comprehensive) ──
_GAP_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)
_PN_RE = re.compile(r"\bPN\b")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}")
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])") 
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')

# ── Reverse mappings: model outputs cleaned format, references use raw format ──
# The training pipeline converted decimal→unicode fractions and Roman→int months,
# so the model outputs ⅓/⅔/½ and "Month 12". Competition references use 0.3333 and "Month XII".
_INT2ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",
              7:"VII",8:"VIII",9:"IX",10:"X",11:"XI",12:"XII"}
_INT_MONTH_RE = re.compile(r"\b([Mm]onth)\s+(\d{1,2})\b")

# Unicode fraction → decimal (reverse of training clean_translation)
_FRAC_TO_DECIMAL = {
    "⅓": "0.3333", "⅔": "0.6666", "½": "0.5",
    "¼": "0.25", "¾": "0.75",
    "⅙": "0.1666", "⅚": "0.8333", "⅝": "0.625",
}
# Also handle e.g. "1⅓" → "1.3333", "2⅔" → "2.6666"
_NUM_FRAC_RE = re.compile(r"(\d+)([⅓⅔½¼¾⅙⅚⅝])")
_STANDALONE_FRAC_RE = re.compile(r"(?<!\d)([⅓⅔½¼¾⅙⅚⅝])")


def postprocess_translation(text: str) -> str:
    """Clean model output to match expected test format."""
    if not text:
        return ""

    # Gap normalization
    text = _GAP_RE.sub("<gap>", text)
    text = _PN_RE.sub("<gap>", text)
    text = _MULTI_GAP_RE.sub("<gap>", text)

    # Curly → straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # En/em dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Ḫ/ḫ → H/h (not in test translations)
    text = text.replace("\u1e2a", "H").replace("\u1e2b", "h")

    # Stray markup
    text = _STRAY_MARKS_RE.sub("", text)

    # ── Reverse training clean_translation() to match raw reference format ──
    # Integer months → Roman numerals (model outputs "Month 12", refs use "Month XII")
    def _int_to_roman_month(m):
        prefix = m.group(1)
        num = int(m.group(2))
        roman = _INT2ROMAN.get(num, str(num))
        return f"{prefix} {roman}"
    text = _INT_MONTH_RE.sub(_int_to_roman_month, text)

    # Unicode fractions → decimal (model outputs ⅓, refs use 0.3333)
    # First handle "1⅓" → "1.3333" patterns
    def _num_frac_to_decimal(m):
        whole = m.group(1)
        frac_char = m.group(2)
        dec = _FRAC_TO_DECIMAL.get(frac_char, "")
        if dec and "." in dec:
            dec_part = dec.split(".")[1]
            return f"{whole}.{dec_part}"
        return m.group(0)
    text = _NUM_FRAC_RE.sub(_num_frac_to_decimal, text)

    # Then standalone fractions: "⅓" → "0.3333"
    def _standalone_frac_to_decimal(m):
        return _FRAC_TO_DECIMAL.get(m.group(1), m.group(1))
    text = _STANDALONE_FRAC_RE.sub(_standalone_frac_to_decimal, text)

    # Strip forbidden chars while preserving <gap>
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(str.maketrans("", "", '\u2014\u2014<>\u2308\u230b\u230a[]+\u02be;'))
    text = text.replace("\x00GAP\x00", " <gap> ")

    # Repetition cleanup
    text = _REPEAT_WORD_RE.sub(r"\1", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

all_predictions = [postprocess_translation(p) for p in all_predictions]

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "translation": all_predictions,
})
submission = submission.sort_values("id").reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved: {OUTPUT_CSV} ({len(submission)} rows)")
print(submission.head())

# %% [markdown]
# ## 7. Sample Translations

# %%
for i in range(min(3, len(submission))):
    row = submission.iloc[i]
    print(f"\n--- Example {row['id']} ---")
    print(f"Translation: {row['translation'][:200]}")
