# MARDUK v7 – Akkadian → English Translation  
# Competition: Deep Past Initiative – Machine Translation (Kaggle)
#
# Multi-model ensemble with MBR (Minimum Bayes Risk) decoding
# Uses public pre-trained ByT5 models + aggressive candidate generation
# + comprehensive pre/post processing based on host recommendations

# %% [markdown]
# # MARDUK v7 – Multi-Model Ensemble × MBR

# %%
import os, gc, re, math, random, logging, warnings
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


def _ngrams(seq, n):
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1))


def sentence_chrf(hyp: str, ref: str, char_order: int = 6,
                  word_order: int = 2, beta: float = 2.0) -> float:
    """Sentence-level chrF++ (0-100 scale)."""
    if not hyp or not ref:
        return 0.0
    pairs = []
    for n in range(1, char_order + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        m = sum(min(h[g], r[g]) for g in h)
        pairs.append((m / max(sum(h.values()), 1), m / max(sum(r.values()), 1)))
    hw, rw = hyp.split(), ref.split()
    for n in range(1, word_order + 1):
        h, r = _ngrams(hw, n), _ngrams(rw, n)
        m = sum(min(h[g], r[g]) for g in h)
        pairs.append((m / max(sum(h.values()), 1), m / max(sum(r.values()), 1)))
    avg_p = sum(p for p, _ in pairs) / len(pairs)
    avg_r = sum(r for _, r in pairs) / len(pairs)
    if avg_p + avg_r == 0:
        return 0.0
    b2 = beta ** 2
    return 100.0 * (1 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


def sentence_bleu(hyp: str, ref: str) -> float:
    """Sentence-level BLEU with add-one smoothing (0-100 scale)."""
    ht, rt = hyp.split(), ref.split()
    if not ht or not rt:
        return 0.0
    log_p = []
    for n in range(1, 5):
        h, r = _ngrams(ht, n), _ngrams(rt, n)
        m = sum(min(h[g], r[g]) for g in h)
        total = max(sum(h.values()), 1)
        log_p.append(math.log((m + 1) / (total + 1)))
    bp = min(1.0, math.exp(1 - len(rt) / max(len(ht), 1)))
    return 100.0 * bp * math.exp(sum(log_p) / len(log_p))

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# %% [markdown]
# ## 1. Configuration

# %%
def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False

def _bf16_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and _cuda_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

# Model paths — multiple public ByT5 models for ensemble diversity
# Each tuple: (path, label, format)
# format="plain" means prefix "translate Akkadian to English: " + transliteration
# format="dualview" means <raw>...</raw><norm>...</norm>
MODEL_CONFIGS = []

_CANDIDATES = [
    # assiaben's optimized ByT5 (used by most top notebooks)
    ("/kaggle/input/final-byt5/byt5-akkadian-optimized-34x", "assiaben-opt34x", "plain"),
    ("/kaggle/input/datasets/assiaben/final-byt5/byt5-akkadian-optimized-34x", "assiaben-opt34x", "plain"),
    # mattiaangeli's MBR v2 (Kaggle Model)
    ("/kaggle/input/byt5-akkadian-mbr-v2/pytorch/default/1", "mbr-v2", "plain"),
    ("/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr-v2/pytorch/default/1", "mbr-v2", "plain"),
    # llkh0a's ByT5 akkadian model
    ("/kaggle/input/byt5-akkadian-model/byt5_akkadian_model", "llkh0a", "plain"),
    ("/kaggle/input/datasets/llkh0a/byt5-akkadian-model/byt5_akkadian_model", "llkh0a", "plain"),
    ("/kaggle/input/byt5-akkadian-model", "llkh0a", "plain"),
    # michaelhernandez4's ByT5 best
    ("/kaggle/input/byt5-akkadian-best/byt5-akkadian-optimized", "mh-best", "plain"),
    ("/kaggle/input/datasets/michaelhernandez4/byt5-akkadian-best/byt5-akkadian-optimized", "mh-best", "plain"),
    ("/kaggle/input/byt5-akkadian-best", "mh-best", "plain"),
    # giovannyrodrguez models
    ("/kaggle/input/modelofinalbyt5/byt5-akkadian-model_final", "giovanny-final", "plain"),
    ("/kaggle/input/datasets/giovannyrodrguez/modelofinalbyt5/byt5-akkadian-model_final", "giovanny-final", "plain"),
    ("/kaggle/input/nomarl36/byt5-akkadian-model_final", "giovanny-norm36", "plain"),
    ("/kaggle/input/datasets/giovannyrodrguez/nomarl36/byt5-akkadian-model_final", "giovanny-norm36", "plain"),
    # manwithacat RAG model
    ("/kaggle/input/byt5-rag-akkadian-v1", "rag-v1", "plain"),
    ("/kaggle/input/datasets/manwithacat/byt5-rag-akkadian-v1", "rag-v1", "plain"),
    # Our own MARDUK model (dual-view format)
    ("/kaggle/input/marduk-byt5-akkadian2english", "marduk-dv", "dualview"),
    ("/kaggle/input/notebooks/theskateborg/marduk-model-download/model", "marduk-dv", "dualview"),
    ("/kaggle/input/marduk-model-download/model", "marduk-dv", "dualview"),
]

# Discover available models
_seen_labels = set()
for path, label, fmt in _CANDIDATES:
    p = Path(path)
    if label not in _seen_labels and p.exists() and (p / "config.json").exists():
        MODEL_CONFIGS.append((str(p), label, fmt))
        _seen_labels.add(label)
        print(f"  Found model: {label} -> {p}")

if not MODEL_CONFIGS:
    # List inputs for debugging
    inp = Path("/kaggle/input")
    if inp.exists():
        for item in sorted(inp.rglob("config.json")):
            print(f"  config.json at: {item.parent}")
    raise FileNotFoundError("No models found!")

print(f"\nUsing {len(MODEL_CONFIGS)} models: {[c[1] for c in MODEL_CONFIGS]}")

# Competition data
for _tp in [
    Path("/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"),
    Path("/kaggle/input/deep-past-initiative-machine-translation/test.csv"),
]:
    if _tp.exists():
        TEST_CSV = _tp
        break
else:
    TEST_CSV = Path("/kaggle/input/deep-past-initiative-machine-translation/test.csv")

OUTPUT_CSV = Path("/kaggle/working/submission.csv")

# %% [markdown]
# ## 2. Generation Configuration

# %%
@dataclass
class MBRConfig:
    max_input_length: int = 512
    max_new_tokens: int = 384
    batch_size: int = 4  # With only 4 test rows, process all at once

    # Beam search
    num_beam_cands: int = 6
    num_beams: int = 12
    length_penalty: float = 1.3
    early_stopping: bool = True
    repetition_penalty: float = 1.2

    # Diverse beam search
    use_diverse_beam: bool = True
    num_diverse_cands: int = 4
    num_diverse_beams: int = 8
    num_beam_groups: int = 4
    diversity_penalty: float = 0.8

    # Multi-temperature sampling
    use_sampling: bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8, 0.95, 1.1])
    num_sample_per_temp: int = 3

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp

    mbr_top_p: float = 0.92
    mbr_pool_cap: int = 64  # Large pool since only 4 samples

    # MBR weights
    mbr_w_chrf: float = 0.55
    mbr_w_bleu: float = 0.25
    mbr_w_jaccard: float = 0.20
    mbr_w_length: float = 0.10

    use_mixed_precision: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16_amp = bool(
            self.use_mixed_precision
            and self.device.type == "cuda"
            and _cuda_bf16_supported()
        )

cfg = MBRConfig()

# %% [markdown]
# ## 3. Preprocessing

# %%
import unicodedata

# --- Preprocessing for "plain" format models ---

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_GAP_UNIFIED_RE = re.compile(
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

def _normalize_gaps(text: str) -> str:
    return _GAP_UNIFIED_RE.sub("<gap>", text)

_CHAR_TRANS = str.maketrans({
    "ḫ":"h","Ḫ":"H","ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})

_DET_UPPER_RE = re.compile(
    r"\(([A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF0-9]{1,6})\)"
)
_DET_LOWER_RE = re.compile(
    r"\(([a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF]{1,4})\)"
)

_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_SUB_X = "ₓ"
_WS_RE = re.compile(r"\s+")

_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}

def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")


def preprocess_plain(text: str) -> str:
    """Preprocess for public ByT5 models (plain format)."""
    text = str(text) if text else ""
    text = _ascii_to_diacritics(text)
    text = _DET_UPPER_RE.sub(r"\1", text)
    text = _DET_LOWER_RE.sub(r"{\1}", text)
    text = _normalize_gaps(text)
    text = text.translate(_CHAR_TRANS)
    text = text.replace(_SUB_X, "")
    text = _KUBABBAR_RE.sub("KÙ.BABBAR", text)
    text = _EXACT_FRAC_RE.sub(_frac_repl, text)
    text = _FLOAT_RE.sub(lambda m: _canon_decimal(float(m.group(1))), text)
    text = _WS_RE.sub(" ", text).strip()
    return "translate Akkadian to English: " + text


# --- Preprocessing for "dualview" format (MARDUK model) ---

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

def _normalize_text_dv(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(AKKADIAN_CHAR_MAP)
    text = text.translate(SUBSCRIPT_MAP)
    text = text.translate(SUPERSCRIPT_MAP)
    # Scribal notations
    text = text.replace("!", "")
    text = re.sub(r"[\u02f9\u2e22\u2e23\u0304]", "", text)
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
    # Line numbers
    text = re.sub(r"(?m)^\s*\d+['\u2019]*[\.\)]\s*", "", text)
    text = text.translate(DASH_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dualview(text: str) -> str:
    """Preprocess for MARDUK dual-view model."""
    source = str(text) if text else ""
    normalized = _normalize_text_dv(source)
    return f"<raw> {source} </raw> <norm> {normalized} </norm>"


# %% [markdown]
# ## 4. Postprocessing (comprehensive, based on host v3 recommendations)

# %%
_PN_RE = re.compile(r"\bPN\b")

# Commodity replacements (only with space-prefixed hyphen)
_COMMODITY_RE = re.compile(r'(?<=\s)-(gold|tax|textiles)\b')
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}

def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]

# Shekel fraction corrections (host confirmed)
_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '\u2153 shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '\u00bd shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

# Grammar markers
_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")

# Quotes
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")

# Months
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12}

def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"

# Repetition cleanup
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")

# Forbidden chars — parentheses KEPT (host confirmed they appear in test)
_FORBIDDEN_TRANS = str.maketrans("", "", '\u2014\u2014<>\u2308\u230b\u230a[]+\u02be;')

# Stray marks
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_EXTRA_STRAY_RE = re.compile(r'(?<!\w)(?:\.\.+|xx+)(?!\w)')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')

# Slash alternatives (protect fractions)
_SLASH_ALT_RE = re.compile(r'(?<![0-9/])\s+/\s+(?![0-9])\S+')

# Hacek normalization (host confirmed not in test)
_HACEK_TRANS = str.maketrans({"\u1e2b": "h", "\u1e2a": "H"})


def postprocess(text: str) -> str:
    """Comprehensive postprocessing matching host v3 recommendations."""
    if not text:
        return ""

    # Gap normalization
    text = _normalize_gaps(text)
    text = _PN_RE.sub("<gap>", text)

    # Commodity replacements
    text = _COMMODITY_RE.sub(_commodity_repl, text)

    # Shekel fractions
    for pat, repl in _SHEKEL_REPLS:
        text = pat.sub(repl, text)

    # Decimal -> fraction
    text = _EXACT_FRAC_RE.sub(_frac_repl, text)
    text = _FLOAT_RE.sub(lambda m: _canon_decimal(float(m.group(1))), text)

    # Grammar markers
    text = _SOFT_GRAM_RE.sub(" ", text)
    text = _BARE_GRAM_RE.sub(" ", text)
    text = _UNCERTAIN_RE.sub("", text)

    # Stray marks
    text = _STRAY_MARKS_RE.sub("", text)
    text = _EXTRA_STRAY_RE.sub("", text)
    text = _SLASH_ALT_RE.sub("", text)

    # Curly -> straight quotes
    text = _CURLY_DQ_RE.sub('"', text)
    text = _CURLY_SQ_RE.sub("'", text)

    # Months
    text = _MONTH_RE.sub(_month_repl, text)

    # Gap dedup
    text = _MULTI_GAP_RE.sub("<gap>", text)

    # Strip forbidden chars while preserving <gap>
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(_FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")

    # Hacek normalization
    text = text.translate(_HACEK_TRANS)

    # Repetition cleanup
    text = _REPEAT_WORD_RE.sub(r"\1", text)
    for n in range(4, 1, -1):
        pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pat, r"\1", text)

    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# %% [markdown]
# ## 5. MBR Selector

# %%
class MBRSelector:
    def __init__(self, pool_cap=64, w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_length=0.10):
        self.pool_cap = pool_cap
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    @staticmethod
    def _chrfpp(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return sentence_chrf(a, b)

    @staticmethod
    def _bleu_score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            return sentence_bleu(a, b)
        except Exception:
            return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 100.0
        if not ta or not tb:
            return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise(self, a: str, b: str) -> float:
        s = (
            self.w_chrf * self._chrfpp(a, b)
            + self.w_bleu * self._bleu_score(a, b)
            + self.w_jaccard * self._jaccard(a, b)
        )
        return s / self._pw_total

    @staticmethod
    def _length_bonus(lengths: List[int], idx: int) -> float:
        if not lengths:
            return 100.0
        median = float(np.median(lengths))
        sigma = max(median * 0.4, 5.0)
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, candidates: List[str]) -> str:
        cands = self._dedup(candidates)
        if self.pool_cap:
            cands = cands[:self.pool_cap]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        lengths = [len(c.split()) for c in cands]
        scores = []

        for i in range(n):
            pw = sum(self._pairwise(cands[i], cands[j]) for j in range(n) if j != i)
            pw /= max(1, n - 1)
            lb = self._length_bonus(lengths, i)
            scores.append(pw + self.w_length * lb)

        best_idx = int(np.argmax(scores))
        return cands[best_idx]

# %% [markdown]
# ## 6. Load Test Data

# %%
test_df = pd.read_csv(TEST_CSV)
print(f"Test set: {len(test_df)} rows")
print(f"Columns: {test_df.columns.tolist()}")

# Sort by text_id + line_start for coherent ordering
if "text_id" in test_df.columns:
    sort_cols = ["text_id"]
    if "line_start" in test_df.columns:
        sort_cols.append("line_start")
    test_df = test_df.sort_values(sort_cols)

sample_ids = test_df["id"].tolist()
raw_transliterations = test_df["transliteration"].fillna("").tolist()

print(f"Sample IDs: {sample_ids}")
for i, t in enumerate(raw_transliterations):
    print(f"  [{i}] {t[:120]}...")

# %% [markdown]
# ## 7. Generate Candidates from All Models

# %%
# Candidate pools: {sample_id: [list of candidate translations]}
candidate_pools = {str(sid): [] for sid in sample_ids}

for model_path, label, fmt in MODEL_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Loading model: {label} ({model_path})")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(cfg.device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params, device={cfg.device}")

    if cfg.device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Prepare inputs based on format
    if fmt == "dualview":
        input_texts = [preprocess_dualview(t) for t in raw_transliterations]
    else:
        input_texts = [preprocess_plain(t) for t in raw_transliterations]

    # Tokenize
    enc = tokenizer(
        input_texts,
        max_length=cfg.max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(cfg.device)

    ctx = _bf16_ctx(cfg.device, cfg.use_bf16_amp)
    B = len(input_texts)

    with torch.inference_mode(), ctx:
        # --- Standard beam search ---
        print(f"  Beam search: {cfg.num_beams} beams -> {cfg.num_beam_cands} candidates")
        beam_out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            do_sample=False,
            num_beams=cfg.num_beams,
            num_return_sequences=cfg.num_beam_cands,
            max_new_tokens=cfg.max_new_tokens,
            length_penalty=cfg.length_penalty,
            early_stopping=cfg.early_stopping,
            repetition_penalty=cfg.repetition_penalty,
            use_cache=True,
        )
        beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

        for i in range(B):
            sid = str(sample_ids[i])
            cands = beam_texts[i * cfg.num_beam_cands : (i + 1) * cfg.num_beam_cands]
            candidate_pools[sid].extend(cands)

        # --- Diverse beam search ---
        if cfg.use_diverse_beam:
            print(f"  Diverse beam: {cfg.num_diverse_beams} beams, {cfg.num_beam_groups} groups -> {cfg.num_diverse_cands} candidates")
            try:
                div_out = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    do_sample=False,
                    num_beams=cfg.num_diverse_beams,
                    num_beam_groups=cfg.num_beam_groups,
                    diversity_penalty=cfg.diversity_penalty,
                    num_return_sequences=cfg.num_diverse_cands,
                    max_new_tokens=cfg.max_new_tokens,
                    length_penalty=cfg.length_penalty,
                    early_stopping=cfg.early_stopping,
                    repetition_penalty=cfg.repetition_penalty,
                    use_cache=True,
                )
                diverse_texts = tokenizer.batch_decode(div_out, skip_special_tokens=True)
                for i in range(B):
                    sid = str(sample_ids[i])
                    cands = diverse_texts[i * cfg.num_diverse_cands : (i + 1) * cfg.num_diverse_cands]
                    candidate_pools[sid].extend(cands)
            except Exception as e:
                print(f"  Diverse beam failed: {e}")

        # --- Multi-temperature sampling ---
        if cfg.use_sampling:
            for temp in cfg.sample_temperatures:
                print(f"  Sampling @ temp={temp:.2f}: {cfg.num_sample_per_temp} candidates")
                try:
                    samp_out = model.generate(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        do_sample=True,
                        num_beams=1,
                        top_p=cfg.mbr_top_p,
                        temperature=temp,
                        num_return_sequences=cfg.num_sample_per_temp,
                        max_new_tokens=cfg.max_new_tokens,
                        repetition_penalty=cfg.repetition_penalty,
                        use_cache=True,
                    )
                    samp_texts = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
                    for i in range(B):
                        sid = str(sample_ids[i])
                        cands = samp_texts[i * cfg.num_sample_per_temp : (i + 1) * cfg.num_sample_per_temp]
                        candidate_pools[sid].extend(cands)
                except Exception as e:
                    print(f"  Sampling @ temp={temp:.2f} failed: {e}")

    # Pool size report
    for sid in sample_ids:
        pool = candidate_pools[str(sid)]
        print(f"  Sample {sid}: {len(pool)} candidates so far")

    # Unload model
    del model, tokenizer, enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"  Model {label} unloaded.")

# %% [markdown]
# ## 8. Postprocess All Candidates

# %%
print("\nPostprocessing candidates...")
for sid in sample_ids:
    sid = str(sid)
    candidate_pools[sid] = [postprocess(c) for c in candidate_pools[sid]]
    unique = len(set(candidate_pools[sid]))
    print(f"  Sample {sid}: {len(candidate_pools[sid])} total, {unique} unique after postprocessing")

# %% [markdown]
# ## 9. MBR Selection

# %%
mbr = MBRSelector(
    pool_cap=cfg.mbr_pool_cap,
    w_chrf=cfg.mbr_w_chrf,
    w_bleu=cfg.mbr_w_bleu,
    w_jaccard=cfg.mbr_w_jaccard,
    w_length=cfg.mbr_w_length,
)

final_translations = {}
for sid in sample_ids:
    sid_str = str(sid)
    pool = candidate_pools[sid_str]
    best = mbr.pick(pool)
    final_translations[sid_str] = best
    print(f"\nSample {sid_str}:")
    print(f"  Pool size: {len(pool)}")
    print(f"  Selected: {best[:200]}")

# %% [markdown]
# ## 10. Create Submission

# %%
submission = pd.DataFrame({
    "id": [str(sid) for sid in sample_ids],
    "translation": [final_translations[str(sid)] for sid in sample_ids],
})
submission = submission.sort_values("id").reset_index(drop=True)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_CSV, index=False)

print(f"\nSubmission saved: {OUTPUT_CSV} ({len(submission)} rows)")
print(submission)

# Final translations
for _, row in submission.iterrows():
    print(f"\n--- {row['id']} ---")
    print(row['translation'])
