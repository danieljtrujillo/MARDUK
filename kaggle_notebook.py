# MARDUK – Akkadian → English Neural Translation
# Competition: Deep Past Initiative – Machine Translation (Kaggle)
#
# This notebook loads a fine-tuned ByT5-base model (v5_detrunc, Score=41.22)
# and generates translations for the test set.
#
# Model: google/byt5-base fine-tuned on 8,480 Akkadian-English pairs with TGT_MAX=256
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

# Generation parameters (grid-searched optimal)
SRC_MAX = 1024
TGT_MAX = 256
NUM_BEAMS = 3
LENGTH_PENALTY = 1.2
NO_REPEAT_NGRAM = 20
BATCH_SIZE = 4

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
# ## 5. Generate Translations

# %%
@torch.no_grad()
def decode_batch(packed_sources: list[str]) -> list[str]:
    inputs = tokenizer(
        packed_sources,
        max_length=SRC_MAX,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        gen_kwargs = dict(
            max_new_tokens=TGT_MAX,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            early_stopping=True,
        )
        if NO_REPEAT_NGRAM > 0:
            gen_kwargs["no_repeat_ngram_size"] = NO_REPEAT_NGRAM
        generated = model.generate(**inputs, **gen_kwargs)

    return tokenizer.batch_decode(generated, skip_special_tokens=True)

all_predictions: list[str] = []
for i in tqdm(range(0, len(packed_sources), BATCH_SIZE), desc="Translating"):
    batch = packed_sources[i : i + BATCH_SIZE]
    preds = decode_batch(batch)
    all_predictions.extend(preds)

print(f"\nGenerated {len(all_predictions)} translations")

# %% [markdown]
# ## 6. Post-Process & Create Submission

# %%
def postprocess_translation(text: str) -> str:
    """Clean model output to match expected test format."""
    # Curly → straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # En/em dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Remove any <big_gap> the model might output
    text = text.replace("<big_gap>", "<gap>")
    # Deduplicate adjacent <gap>
    text = re.sub(r"(<gap>)(?:[\s\-]*<gap>)+", r"\1", text)
    # Ḫ/ḫ → H/h (not in test translations)
    text = text.replace("\u1e2a", "H").replace("\u1e2b", "h")
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
