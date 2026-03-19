# MARDUK – Akkadian → English Neural Translation  (v15 – fast batched beam)
# Competition: Deep Past Initiative – Machine Translation (Kaggle)
#
# v15 changes from v14:
#   - Removed MBR self-ensemble (was causing timeout on hidden test set)
#   - Batched beam search for ~10x speedup
#   - FP16 model loading for 2x memory/speed gain
#   - Length-sorted batching to minimise padding
#   - Time budget with automatic greedy fallback
#   - OOM-safe: halves batch size on CUDA OOM

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

# Generation parameters
SRC_MAX = 1024
TGT_MAX = 512
NUM_BEAMS = 5
BATCH_SIZE = 8          # inputs per forward pass (halved automatically on OOM)
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
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_PATH), torch_dtype=torch.float16
).to(device)
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
    """Translate a batch of packed source strings."""
    inputs = tokenizer(
        sources,
        max_length=SRC_MAX,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=TGT_MAX,
        num_beams=num_beams,
        early_stopping=True,
    )
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
