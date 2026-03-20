# MARDUK – Akkadian → English Neural Translation  (v20)
# Competition: Deep Past Initiative – Machine Translation (Kaggle)
#
# v20: Restore v5 params (scored 26.2) + enhanced post-processing
#   - DUAL-VIEW format: "<raw> ... </raw> <norm> ... </norm>"
#   - Generation: v5 params (SRC_MAX=1024, TGT_MAX=1024, NUM_BEAMS=3, LP=1.2)
#   - bfloat16 autocast (v5 used this; FP32 was a v19 regression)
#   - NO repetition_penalty (v5 didn't use it)
#   - Enhanced post-processing: decimal→fraction, ḫ→h, gap cleanup
#   - Batched beam search + time budget + OOM-safe batch halving

# %% [markdown]
# # MARDUK – Akkadian to English Translation

# %%
import os
import re
import time
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

# Generation parameters — restored from v5 (scored 26.2 on LB)
SRC_MAX = 1024
TGT_MAX = 1024
NUM_BEAMS = 3            # v5 used 3 (not 5)
LENGTH_PENALTY = 1.2     # v5 used 1.2 (not 1.3)
NO_REPEAT_NGRAM = 20     # safety: prevent byte-level degeneration on unseen test
BATCH_SIZE = 4          # inputs per forward pass (halved automatically on OOM)
TIME_LIMIT = 8 * 3600   # 8 hours hard wall (1 h safety margin on 9 h limit)

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
# v5 VOWEL_MAP: convert diacritics to ASCII numbers (matches v5_detrunc training data)
VOWEL_MAP = {
    "\u00e1": "a2", "\u00e0": "a3", "\u00e9": "e2", "\u00e8": "e3",
    "\u00ed": "i2", "\u00ec": "i3", "\u00fa": "u2", "\u00f9": "u3",
}

def normalize_akkadian_chars(text: str) -> str:
    text = text.translate(AKKADIAN_CHAR_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    for src, dst in VOWEL_MAP.items():
        text = text.replace(src, dst)
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
    """Create dual-view input (matches training format in train_prepared.csv)."""
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
# ## 5. Generate Translations (Batched Beam Search)

# %%
@torch.no_grad()
def generate_batch(sources, num_beams=NUM_BEAMS):
    """Translate a batch — bfloat16 autocast (matches v5 which scored 26.2)."""
    inputs = tokenizer(
        sources,
        max_length=SRC_MAX,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    gen_kwargs = dict(
        max_new_tokens=TGT_MAX,
        num_beams=num_beams,
        length_penalty=LENGTH_PENALTY,
        early_stopping=True,
    )
    if NO_REPEAT_NGRAM > 0:
        gen_kwargs["no_repeat_ngram_size"] = NO_REPEAT_NGRAM
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Sort by input length for efficient padding, then unsort after
order = sorted(range(len(packed_sources)), key=lambda i: len(packed_sources[i]))
sorted_sources = [packed_sources[i] for i in order]

all_predictions_sorted = [""] * len(sorted_sources)
batch_size = BATCH_SIZE
t0 = time.time()

i = 0
pbar = tqdm(total=len(sorted_sources), desc="Translating")
while i < len(sorted_sources):
    elapsed = time.time() - t0
    remaining_examples = len(sorted_sources) - i

    # Switch to greedy if running low on time
    beams = NUM_BEAMS
    if elapsed > TIME_LIMIT * 0.85 and remaining_examples > 50:
        beams = 1  # greedy fallback
        batch_size = max(batch_size, 16)
        if not hasattr(generate_batch, "_warned_greedy"):
            print(f"\n⚠ Time budget tight ({elapsed/3600:.1f}h), switching to greedy")
            generate_batch._warned_greedy = True

    batch_end = min(i + batch_size, len(sorted_sources))
    batch = sorted_sources[i:batch_end]

    try:
        preds = generate_batch(batch, num_beams=beams)
        all_predictions_sorted[i:batch_end] = preds
        pbar.update(batch_end - i)
        i = batch_end
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 1:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"\n⚠ OOM – reducing batch_size to {batch_size}")
        else:
            raise
pbar.close()

# Unsort back to original order
all_predictions = [""] * len(packed_sources)
for orig_idx, pred in zip(order, all_predictions_sorted):
    all_predictions[orig_idx] = pred

elapsed = time.time() - t0
print(f"\nTranslated {len(all_predictions)} examples in {elapsed/60:.1f} min "
      f"({elapsed/len(all_predictions):.2f} s/example)")

# %% [markdown]
# ## 6. Post-Process & Create Submission

# %%
# Decimal → unicode fraction map (host confirmed: test uses ONLY unicode fractions)
DECIMAL_TO_FRACTION = [
    ("0.8333", "\u215a"),  # ⅚
    ("0.6666", "\u2154"),  # ⅔
    ("0.3333", "\u2153"),  # ⅓
    ("0.1666", "\u2159"),  # ⅙
    ("0.625", "\u215d"),   # ⅝
    ("0.75", "\u00be"),    # ¾
    ("0.25", "\u00bc"),    # ¼
    ("0.5", "\u00bd"),     # ½
]

def postprocess_translation(text: str) -> str:
    """Clean model output to match expected test format.
    
    Based on competition host guidance and discussion insights:
    - Test uses unicode fractions (not decimals)
    - No ḫ/Ḫ in test translations
    - Straight quotes only
    - No scribal annotations (fem., sing., pl., etc.)
    - Single <gap> markers (no duplicates/big_gap)
    """
    if not text:
        return ""
    # Curly → straight quotes (host confirmed: dead quotes, no curl)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # En/em dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Ḫ/ḫ → H/h (host confirmed: neither appears in test)
    text = text.replace("\u1e2a", "H").replace("\u1e2b", "h")
    # <big_gap> → <gap>
    text = text.replace("<big_gap>", "<gap>")
    # Deduplicate adjacent <gap>
    text = re.sub(r"(<gap>)(?:[\s\-]*<gap>)+", r"\1", text)
    # Decimal → unicode fraction (host confirmed: test has only unicode fractions)
    for dec, frac in DECIMAL_TO_FRACTION:
        text = text.replace(dec, frac)
    # Remove scribal annotations not in test (host confirmed)
    text = re.sub(r"\bfem\.\s*", "", text)
    text = re.sub(r"\bsing\.\s*", "", text)
    text = re.sub(r"\bpl\.\s*", "", text)
    text = re.sub(r"\bplural\b\s*", "", text)
    # Remove stray (?) — host confirmed not in test
    text = text.replace("(?)", "")
    # Subscript digits → normal digits in output
    text = text.translate(str.maketrans(
        "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089",
        "0123456789",
    ))
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
