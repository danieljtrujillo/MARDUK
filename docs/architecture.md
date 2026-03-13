# Architecture

## Problem framing

Task: translate **Old Assyrian transliteration** to **English**.

The scaffold treats the source as:
- noisy and symbol-heavy
- partly formulaic
- sensitive to names, numbers, determinatives, and damage markers

The system therefore separates:
- a strong off-the-shelf seq2seq baseline
- a source-heavy hybrid model intended to improve behavior on long inputs and structured spans

## Baseline path

Use `ByT5`, `mT5`, or `T5` through Hugging Face `AutoModelForSeq2SeqLM`.

Why:
- cheap to train
- stable generation stack
- straightforward beam search
- good control for whether the hybrid is actually helping

## Hybrid path

### Source side
- byte-level source vocabulary
- optional raw + normalized dual view
- encoder wrapper:
  - `mamba_ssm` if installed
  - bi-GRU fallback otherwise

### Target side
- Transformer decoder
- tokenized English target
- cross-attention over encoder states

### Auxiliary objectives
Attached to encoder hidden states:
- name BIO tagging
- number BIO tagging
- damage/uncertainty BIO tagging
- optional normalization reconstruction

## Input packing

```text
<meta_period=...> <meta_genre=...> <meta_provenance=...> <raw> ... </raw> <norm> ... </norm>
```

You can switch off metadata or normalized view in config.

## Why this split

The source side is where long, irregular symbolic sequences show up. The target side is normal English generation. If the source encoder helps, it should show up first in:
- name span recall
- number exact match
- length-bucket robustness
- fewer hallucinations around damage markers

If it does not, stop the SSM line and keep the baseline.
