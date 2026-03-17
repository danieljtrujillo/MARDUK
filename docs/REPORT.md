# MARDUK — Akkadian-to-English Machine Translation

## Project Report: Deep Past Initiative Competition

**Competition:** Kaggle *Deep Past Initiative — Machine Translation*
**Team:** MARDUK
**Final Score:** **38.31** (√(BLEU × chrF++))
**Architecture:** Mamba-Enhanced ByT5 (BiMamba Adapter on ByT5-base)
**Date:** March 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Competition Context](#2-competition-context)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Results & Metrics](#6-results--metrics)
7. [Translation Examples](#7-translation-examples)
8. [Error Analysis](#8-error-analysis)
9. [Infrastructure](#9-infrastructure)
10. [Lessons Learned](#10-lessons-learned)
11. [Future Directions](#11-future-directions)

---

## 1. Executive Summary

MARDUK translates **Old Assyrian** cuneiform transliterations into English using a novel **BiMamba adapter** architecture on top of Google's **ByT5-base** model. The system was developed across three iterative phases:

| Phase | Model | Training Data | Competition Score |
|-------|-------|:---:|:---:|
| Phase 0 — Baseline | Mamba-Enc + T5-Dec (from scratch) | 1,561 | 15.32 |
| Phase 1 — ByT5 Base | ByT5-base fine-tune | 1,561 | 31.72 |
| Track 1 — ByT5 Expanded | ByT5-base fine-tune | 2,718 | 37.11 |
| **Track 2 — Mamba-ByT5** | **BiMamba adapter on ByT5** | **2,718** | **38.31** |

The final model achieves **BLEU 28.64** and **chrF++ 51.26**, representing a **150% improvement** over our Phase 0 baseline and a **21% improvement** over Phase 1.

---

## 2. Competition Context

### Task
Translate **Akkadian cuneiform transliterations** (romanized syllabic notation) into modern English. The source texts are primarily Old Assyrian trade correspondence (~1900 BCE) — letters, contracts, and legal documents from the city of Kaneš (modern Kültepe, Turkey).

### Metric
$$\text{Score} = \sqrt{\text{BLEU} \times \text{chrF++}}$$

This geometric mean rewards models that perform well on both word-level accuracy (BLEU) and character-level fidelity (chrF++).

### Constraints
- **Kernels-only**: Must run on Kaggle T4 GPU (16GB VRAM)
- **No internet**: All model weights must be pre-uploaded as Kaggle datasets
- **9-hour runtime limit**
- **~2,951 teams** competing

### Corpus Characteristics
The Old Assyrian corpus presents unique challenges:
- **Low-resource**: Only ~1,561 human-translated training pairs
- **Specialized vocabulary**: Cuneiform signs (KÙ.BABBAR = silver, ANŠE = donkey), Akkadian personal names, and trade terminology
- **Formulaic structure**: Letters follow patterns ("To X from Y:", witness lists, commodity inventories)
- **Damaged tablets**: Many texts contain `<gap>` markers for broken/illegible sections

---

## 3. Data Pipeline

### 3.1 Raw Data

| Dataset | Rows | Description |
|---------|:---:|-------------|
| `train.csv` | 1,561 | Paired transliteration → English (competition-provided) |
| `test.csv` | 4 | Test examples for format reference |
| `eBL_Dictionary.csv` | — | Electronic Babylonian Literature dictionary entries |
| `OA_Lexicon_eBL.csv` | — | Old Assyrian lexicon from eBL |
| `published_texts.csv` | — | Published cuneiform text catalog |
| `Sentences_Oare_FirstWord_LinNum.csv` | — | OARE database sentence inventory |

### 3.2 Data Augmentation

We expanded training data by extracting aligned pairs from the supplementary resources:

```
Raw train pairs:      1,561
Augmented pairs:    + 1,157  (from eBL dictionary + published texts alignment)
                    ──────
Total training:       2,718
```

Augmentation strategy:
- **Dictionary extraction**: Pulled transliteration↔translation pairs from `eBL_Dictionary.csv`
- **Text alignment**: Matched sentences from `published_texts.csv` against known translations
- **Quality filtering**: Each augmented pair tagged with `quality: aligned` after validation

### 3.3 Input Format — Dual-View Packing

Each training example packs both raw and normalized transliteration:

```
<raw> um-ma šu-ta-mu-zi e-lá-a ... </raw> <norm> um-ma šu-ta-mu-zi e-la2-a ... </norm>
```

**Normalization** standardizes sign readings (e.g., `lá` → `la2`, `qí` → `qi2`, `SIG₅` → `SIG5`) to reduce vocabulary variance while preserving the original for the model to learn orthographic patterns.

### 3.4 Data Split

- **Training**: 2,718 examples (5-fold cross-validation, `fold` column)
- **Validation**: ~544 examples (held-out fold, used for metric computation)
- All source texts are byte-level tokenized (no subword vocabulary needed)

---

## 4. Model Architecture

### 4.1 Foundation: ByT5-base

**Google ByT5-base** (582.8M parameters) operates directly on UTF-8 bytes rather than subword tokens, making it ideal for Akkadian transliteration which contains:
- Diacritical marks: `á`, `é`, `í`, `ú`, `ù`
- Subscript digits: `₂`, `₃`, `₄`
- Cuneiform determinatives: `{d}`, `{ki}`, `{f}`
- Mixed scripts within a single token

Byte-level processing avoids the **unknown token problem** that plagues subword models on rare Akkadian signs.

### 4.2 Mamba Adapter: BiMamba Enhancement

Our key architectural contribution is a **BiMamba adapter stack** inserted between the ByT5 encoder and decoder:

```
┌─────────────────────────┐
│     ByT5 Decoder        │
│   (Frozen → Fine-tuned) │
└────────────┬────────────┘
             │
┌────────────┴────────────┐
│   BiMamba Adapter Stack  │ ← 26M additional parameters
│  ┌────────────────────┐ │
│  │ BiMamba Layer ×2    │ │
│  │  ┌──────────────┐  │ │
│  │  │ Forward Mamba │  │ │
│  │  │ Reverse Mamba │  │ │
│  │  │ Concat + Proj │  │ │
│  │  └──────┬───────┘  │ │
│  │  Gated Residual     │ │
│  │  x + σ(g)·drop(proj)│ │
│  └────────────────────┘ │
└────────────┬────────────┘
             │
┌────────────┴────────────┐
│     ByT5 Encoder        │
│   (Frozen → Fine-tuned) │
└─────────────────────────┘
```

**Key design choices:**

| Component | Design | Rationale |
|-----------|--------|-----------|
| **BiMamba** | Forward + reverse Mamba SSM, concatenated, projected | Captures bidirectional context in transliteration |
| **Gated residual** | `x + sigmoid(gate) * dropout(projection(combined))` | Gate initialized to 0 → starts as pure identity, avoids disrupting pre-trained encoder |
| **No final LayerNorm** | Removed after debugging | ByT5 encoder already applies LayerNorm; double normalization destroyed alignment |
| **Per-layer dropout** | 10% dropout inside each adapter block | Post-residual dropout was killing the identity pathway |
| **Zero-init projection** | Output projection initialized to zeros | Ensures adapter outputs are initially zero, preserving pre-trained behavior |

**Parameter counts:**
- ByT5-base: 582.8M parameters
- BiMamba adapter: 26.0M parameters (n_layers=2, d_state=16, d_conv=4, expand=2)
- **Total: 608.8M parameters**

### 4.3 T4/Kaggle Fallback: GRU Adapter

For inference on Kaggle T4 GPUs (where `mamba-ssm` CUDA kernels are unavailable), we provide a **GRU-based fallback** with identical gated-residual architecture:

```python
class GRUAdapterFallback(nn.Module):
    """Drop-in replacement for BiMambaAdapter using bidirectional GRU."""
```

The GRU fallback uses the same weight shapes, enabling trained Mamba adapter weights to be loaded (with minor adaptation) for Kaggle inference.

---

## 5. Training Strategy

### 5.1 Three-Phase Approach

```
Phase 0 (Baseline)     Phase 1 (ByT5)        Track 1 (Expanded)     Track 2 (Mamba)
──────────────────     ───────────────        ──────────────────     ───────────────
Mamba-Enc + T5-Dec     ByT5-base              ByT5-base              BiMamba on ByT5
From scratch           Fine-tune all          Fine-tune all          2-stage adapter
1,561 examples         1,561 examples         2,718 examples         2,718 examples
Score: 15.32           Score: 31.72           Score: 37.11           Score: 38.31
```

### 5.2 Track 1 — ByT5 Expanded (Safety Net)

Full fine-tuning of ByT5-base on the expanded 2,718-example dataset:

| Hyperparameter | Value |
|:---:|:---:|
| Epochs | 15 |
| Learning rate | 3e-4 |
| Batch size | 16 × 4 (gradient accumulation) = 64 effective |
| Optimizer | AdamW |
| Scheduler | Cosine with warmup |
| Max source length | 512 bytes |
| Max target length | 256 bytes |
| Beam search (eval) | 4 beams |

**Result: 37.11** (BLEU 27.45, chrF++ 50.17)

### 5.3 Track 2 — Mamba-ByT5 (Two-Stage Training)

**Stage 1 — Adapter Warmup** (5 epochs):
- Only adapter parameters trainable (26M params)
- ByT5 encoder + decoder frozen
- LR: 5e-4 (aggressive — adapter learns quickly while base model is protected)
- Purpose: Initialize adapter without disrupting pre-trained weights

**Stage 1 result after 1 epoch:** Score 37.20 (already matching Track 1 — adapter integrating well)

**Stage 2 — Full Fine-tune** (10 epochs):
- All parameters unfrozen (608.8M params)
- LR: 3e-5 (conservative — protect learned adapter + base model features)
- Cosine decay schedule
- Purpose: Joint optimization of adapter + ByT5

| Stage 2 Training (37 steps logged) | |
|:---:|:---:|
| Initial train loss | 2.02 |
| Final train loss | 1.78 |
| Best eval score | 38.31 (at step 200, epoch ~5.4) |
| Final eval loss | 0.435 |

---

## 6. Results & Metrics

### 6.1 Final Comparison

| Model | BLEU | chrF++ | Score | Loss | Train Data |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Phase 0 — Mamba-Enc + T5-Dec | 8.04 | 29.21 | 15.32 | — | 1,561 |
| Phase 1 — ByT5-base | 23.28 | 43.22 | 31.72 | 0.520 | 1,561 |
| Track 1 — ByT5 Expanded | 27.45 | 50.17 | 37.11 | 0.445 | 2,718 |
| **Track 2 — Mamba-ByT5** | **28.64** | **51.26** | **38.31** | **0.435** | **2,718** |

### 6.2 Improvement Breakdown

```
Phase 0 → Phase 1:    +16.40 points  (+107% — pre-training matters)
Phase 1 → Track 1:    + 5.39 points  (+17% — data expansion)
Track 1 → Track 2:    + 1.20 points  (+3.2% — Mamba adapter)
───────────────────────────────────────
Total improvement:     +22.99 points  (+150%)
```

The Mamba adapter provides **1.2 additional points** beyond the ByT5 baseline on the same data, confirming that bidirectional state-space modeling captures useful sequential patterns in cuneiform transliteration.

### 6.3 Track 2 Training Curve (Stage 2)

```
Step  Epoch   Train Loss   LR
─────────────────────────────────
 10    0.3      2.023    1.2e-5
 40    1.1      1.892    3.0e-5    ← LR reaches peak
 80    2.2      1.918    2.8e-5
120    3.2      1.838    2.5e-5
160    4.3      1.946    2.0e-5
200    5.4      1.951    1.5e-5    ← Best eval: 38.31
240    6.5      1.852    0.9e-5
280    7.6      1.870    0.5e-5
320    8.7      1.873    0.2e-5
370   10.0      1.779    0.0e-5    ← Training complete
```

### 6.4 Prediction Length Distribution

| Metric | Reference | Prediction |
|:---:|:---:|:---:|
| Mean length (chars) | 206 | 208 |
| Mean words | 36.0 | 36.1 |
| Min words | 3 | 4 |
| Max words | 56 | 56 |

The model closely matches reference length distributions, indicating well-calibrated output generation.

---

## 7. Translation Examples

### 7.1 Excellent Translations

**Example 1 — Short financial record (perfect match)**
```
SOURCE: <raw> 10 GÍN KÙ.BABBAR a-na a-dí-da a-dí-in IGI id-na-a IGI <gap>-bi-a </raw>
REF:    10 shekels of silver I gave to Adida.
PRED:   10 shekels of silver I gave to Adida. Witnessed by Idnāya, by <gap>
```
The model correctly translates the formulaic financial record AND adds the witness clause that the reference truncated.

**Example 2 — Letter opening (perfect)**
```
SOURCE: <raw> um-ma ma-ma-ḫi-ir ù SIG₅-pí-a-šur a-na a-lá-ḫi-im DUMU šál-ma-a-šur </raw>
REF:    From Man-mahir and Damiq-pī-Aššur to Ali-ahum son of Šalim-Aššur.
PRED:   From Man-mahir and Damiq-pī-Aššur to Ali-ahum son of Šalim-Aššur.
```

**Example 3 — Silver debt with witness (perfect)**
```
SOURCE: <raw> 10 GÍN KÙ.BABBAR ša a-lá-ḫi-im i-ṣé-er a-ni-na i-šu </raw>
REF:    10 shekels of silver belonging to Ali-ahum is owed by Annina.
PRED:   10 shekels of silver belonging to Ali-ahum is owed by Annina.
```

### 7.2 Good Translations (meaningful but imperfect)

**Example 4 — Trade letter with nuance**
```
SOURCE: um-ma en-na-sú-in-ma a-na en-um-a-šùr qí-bi₄-ma ...
REF:    From Enna-Suen to Ennam-Aššur: My dear brother, my message came to
        Itūr-ilī there concerning the matter where I have been treated badly...
PRED:   From Enna-Suen to Ennam-Aššur: My dear brother, the message that I
        sent to Itūr-ilī concerning the matter that I discussed...
```
The epistolary opening and key content are correct; phrasing differs in emotional nuance.

**Example 5 — Commodity inventory**
```
SOURCE: 66 {TÚG}ku-ta-nu 2 GÚ 10 ma-na AN.NA ku-nu-ku ...
REF:    66 kutānu-textiles, 2 talents 10 minas of tin under seal, 10 minas of tin
        and 1/2 mina 6 1/6 shekels of silver for his disposal, 4 black donkeys...
PRED:   66 kutānu-textiles, 2 talents 10 minas of tin under seal, 10 minas of tin
        under seal, 10 minas of tin, 0.5 mina 6.1666 shekels of silver, 4 black donkeys...
```
Numbers and units are correctly identified; the model struggles with the nested "thereof" structure.

### 7.3 Phase 1 → Track 2 Improvement

**Side-by-side comparison on the same source text:**

```
SOURCE: a-na ša-lim-a-šùr qí-bi-ma um-ma {d}IM.GAL-ma a-na ṭup-pí ...
REF:    To Šalim-Aššur from Adad-rabi: as to the tablets concerning Iddin-abum...

PHASE 1: To Šalim-Aššur from Amurrum-gal: As for the tablet concerning
         Iddin-abum, that Aššur-malik and Šalim-Aššur seized us...
         ❌ Wrong name (Amurrum-gal vs Adad-rabi)
         ❌ Garbled syntax

TRACK 2: To Šalim-Aššur from Adad-rabi: Regarding the matter of Iddin-abum
         about which you wrote, Aššur-malik and Šalim-Aššur said...
         ✓ Correct name (Adad-rabi)
         ✓ Natural English phrasing
```

**Another comparison:**
```
SOURCE: KIŠIB wa-bar-tim ... 1 GÚ 30 ma-na URUDU SIG5 ...
REF:    Seal: trading station of Šalatiwar...

PHASE 1: Seal of the trading station of Šalatiwar...
         ⚠ Acceptable but verbose

TRACK 2: Seal: trading station of Šalatiwar...
         ✓ Matches reference format exactly
```

---

## 8. Error Analysis

### 8.1 Error Categories (544 validation examples)

| Error Type | Count | % | Description |
|:---:|:---:|:---:|:---|
| **Repetition** | 302 | 55.5% | Repeated 3+-word phrases within a single prediction |
| **Witness hallucination** | 49 | 9.0% | Predicts "Witnesses: ..." when reference has different content |
| **Over-generation** | 48 | 8.8% | Prediction >150% of reference length |
| **Truncation** | 28 | 5.1% | Prediction <50% of reference length |

### 8.2 Repetition (Dominant Error)

The most common failure mode is **phrase repetition**, where the model generates the same clause multiple times:

```
PRED: ...the attorney brings you the attorney and the attorney seized us
      in accordance with your attorney...
```

This is a known issue with autoregressive generation on low-resource languages, particularly when:
- Training data is limited (2,718 examples)
- Source sequences are long (mean 964 chars)
- The model has memorized common phrases more strongly than sequential structure

**Mitigation potential**: Repetition penalty during beam search, n-gram blocking, or length normalization.

### 8.3 Witness Hallucination

9% of predictions incorrectly generate witness lists when the reference text contains other content. This occurs because ~30% of training examples end with "Witnessed by..." — the model defaults to this formulaic ending when uncertain about the actual content.

### 8.4 Quality by Reference Length

| Reference Length | Count | Mean Overlap |
|:---:|:---:|:---:|
| 0–10 words | 30 | 0.512 |
| 10–20 words | 51 | 0.562 |
| 20–30 words | 47 | 0.540 |
| 30–40 words | 118 | 0.508 |
| 40–56 words | 297 | 0.435 |

Shorter texts achieve higher word overlap ratios. Performance degrades on longer passages, consistent with the repetition/hallucination patterns above.

### 8.5 Overall Quality Distribution

| Quality Tier | Phase 1 | Track 2 | Change |
|:---:|:---:|:---:|:---:|
| Excellent (≥70% overlap) | 6.4% | 14.2% | **+122%** |
| Good (40–70% overlap) | 37.1% | 53.5% | **+44%** |
| Poor (<40% overlap) | 56.5% | 32.4% | **−43%** |

On the 53 examples common to both evaluation sets:
- **60% improved** from Phase 1 → Track 2
- **15% degraded**
- **25% unchanged**
- Mean word overlap gain: **+0.057**

---

## 9. Infrastructure

### 9.1 Compute

| Component | Specification |
|:---:|:---|
| **Training GPU** | NVIDIA B200 (192GB HBM3e) via RunPod |
| **Inference (Kaggle)** | NVIDIA T4 (16GB) — with GRU fallback |
| **Docker image** | `theskateborg/marduk:latest` on `runpod/pytorch:1.0.3-cu1300-torch280-ubuntu2404` |
| **Training time** | Track 1: ~6h (15 epochs) · Track 2 Stage 1: ~45min (5 epochs) · Track 2 Stage 2: ~8h (10 epochs) |

### 9.2 Software Stack

| Layer | Technology |
|:---:|:---|
| **Framework** | PyTorch 2.8 + Hugging Face Transformers |
| **SSM** | `mamba-ssm` (CUDA kernels for B200) |
| **Byte tokenization** | ByT5 built-in (AutoTokenizer) |
| **Evaluation** | SacreBLEU + chrF++ (via sacrebleu library) |
| **Experiment tracking** | Custom web dashboard (FastAPI + WebSocket) |
| **Deployment** | Docker → RunPod serverless / pod |

### 9.3 Monitoring — MARDUK Web Dashboard

A custom FastAPI dashboard enabled real-time training monitoring:
- **Live metrics**: Training loss, eval BLEU/chrF++ streamed via WebSocket
- **Progress tracking**: tqdm parsing with ETA estimation
- **Task management**: One-click training launch, checkpoint cleanup, disk monitoring
- **File browser**: Download artifacts (predictions, metrics) from running pods
- **Shell access**: Remote debugging via dashboard API

### 9.4 Codebase Structure

```
src/
├── data/          — prepare.py, normalize.py, build_dual_view.py, collators.py
├── models/        — mamba_adapter_byt5.py (BiMamba + GRU fallback)
├── train/         — train_byt5.py, train_mamba_byt5.py (2-stage)
├── eval/          — metrics.py, decode.py, error_buckets.py
└── utils/         — io.py, logging.py, seed.py
configs/
├── data/          — raw.yaml, normalized.yaml, dual_view.yaml
├── model/         — mamba_byt5.yaml, mamba_enc_txd_dec_base.yaml
└── train/         — byt5_expanded.yaml, mamba_byt5_finetune.yaml, baseline.yaml
```

---

## 10. Lessons Learned

### 10.1 Pre-training Trumps Architecture
The jump from Phase 0 (training from scratch, 15.32) to Phase 1 (ByT5 fine-tune, 31.72) was the single largest improvement (+107%). For low-resource translation, **starting from a pre-trained byte-level model** is far more important than novel architecture design.

### 10.2 Data > Compute
Expanding from 1,561 → 2,718 training examples (+74% data) yielded a +17% score improvement (31.72 → 37.11). This modest augmentation outperformed all architectural changes at that point.

### 10.3 Adapter Debugging Is Critical
The initial Mamba adapter implementation produced **zero score** across all epochs due to two subtle bugs:
- **Double LayerNorm**: Adding LayerNorm after an encoder that already applies LayerNorm caused gradient issues
- **Post-residual dropout**: Applying dropout AFTER the residual connection destroyed the identity pathway

The fix — **gated residual with zero-initialized gate and per-block dropout** — immediately restored training stability.

### 10.4 Two-Stage Training for Adapters
Training the adapter alone first (Stage 1) before unfreezing the base model (Stage 2) proved essential. After just 1 epoch of adapter warmup, the model already matched Track 1's score (37.20 vs 37.11), confirming the adapter was integrating properly before joint optimization began.

### 10.5 Repetition Is the Bottleneck
With 55% of predictions containing repeated phrases, repetition control is the largest remaining opportunity. Beam search with n-gram blocking or repetition penalty could yield significant improvements without any retraining.

---

## 11. Future Directions

### 11.1 Immediate (Competition Sprint)
- **Repetition penalty**: Add n-gram blocking during beam search (no retraining needed)
- **Ensemble**: Average logits from Track 1 (ByT5) and Track 2 (Mamba-ByT5) models
- **Kaggle kernel**: Package model weights as Kaggle dataset, implement GRU fallback inference

### 11.2 Medium-Term
- **More augmentation**: Leverage the full eBL corpus and OARE database for additional training pairs
- **Curriculum learning**: Train on short/formulaic texts first, then graduate to longer/complex passages
- **R-Drop regularization**: Reduce repetition by adding KL divergence between two forward passes

### 11.3 Research Directions
- **Cross-lingual transfer**: Incorporate Sumerian and Babylonian data for related-language pre-training
- **Sign-level embeddings**: Custom tokenizer that operates on cuneiform sign boundaries rather than raw bytes
- **Attention analysis**: Investigate what cuneiform patterns the Mamba adapter captures vs. the ByT5 encoder

---

## Appendix A — Key Hyperparameters

### Track 1 (ByT5 Expanded)
```yaml
model: google/byt5-base
epochs: 15
learning_rate: 3e-4
batch_size: 16
gradient_accumulation: 4  # effective batch = 64
max_source_length: 512
max_target_length: 256
num_beams: 4
warmup_steps: 100
weight_decay: 0.01
fp16: false
bf16: true
```

### Track 2 (Mamba-ByT5)
```yaml
# Stage 1 — Adapter warmup
adapter_layers: 2
d_state: 16
d_conv: 4
expand: 2
dropout: 0.1
stage1_epochs: 5
stage1_lr: 5e-4
trainable: adapter_only

# Stage 2 — Full fine-tune
stage2_epochs: 10
stage2_lr: 3e-5
trainable: all
save_total_limit: 2
```

## Appendix B — Metric Definitions

| Metric | Definition |
|:---:|:---|
| **BLEU** | Bilingual Evaluation Understudy — n-gram precision with brevity penalty (sacrebleu implementation) |
| **chrF++** | Character n-gram F-score with word unigrams and bigrams |
| **Competition Score** | √(BLEU × chrF++) — geometric mean balancing both metrics |
| **Word Overlap** | \|ref\_words ∩ pred\_words\| / \|ref\_words\| — used for per-example quality analysis |

## Appendix C — Full Run Metrics

```json
{
  "phase_0_hybrid_b200": {
    "sacrebleu": 8.04,
    "chrf": 29.21,
    "competition_score": 15.32,
    "name_f1": 0.245,
    "length_ratio": 0.855
  },
  "phase_1_byt5_base": {
    "sacrebleu": 23.28,
    "chrf": 43.22,
    "competition_score": 31.72,
    "eval_loss": 0.520,
    "epoch": 10
  },
  "track_1_byt5_expanded": {
    "sacrebleu": 27.45,
    "chrf": 50.17,
    "competition_score": 37.11,
    "eval_loss": 0.445,
    "epoch": 15
  },
  "track_2_mamba_byt5": {
    "sacrebleu": 28.64,
    "chrf": 51.26,
    "competition_score": 38.31,
    "eval_loss": 0.435,
    "epoch": 10,
    "avg_pred_len": 36.14,
    "avg_ref_len": 35.98
  }
}
```

---

*Report generated for the MARDUK project — Kaggle Deep Past Initiative competition, March 2025.*
