# MARDUK
### **M**amba-**A**ugmented **R**econstruction & **D**ecoding of **U**nknown **K**uneiform

Neural machine translation for the Deep Past transliteration-to-English task.

This repo gives you:

- a reproducible **baseline** path with Hugging Face seq2seq models (`ByT5`, `mT5`, `T5`)
- a **hybrid** path for a byte-level source encoder plus Transformer decoder
- data normalization, fold generation, evaluation, and error-bucket reporting
- Markdown docs for architecture, experiments, data contract, and runbook

The baseline path is intended to be runnable once the dataset is placed in `data/raw/` and the column mapping is configured.

The hybrid path is a real code scaffold with:
- byte-level source encoding
- Transformer decoder
- auxiliary span heads for names, numbers, and damage markers
- an encoder wrapper that will use `mamba_ssm` if installed, otherwise falls back to a bi-GRU so the training loop can still execute

## Expected data layout

Put your competition CSV files here:

```text
data/raw/
  train.csv
  test.csv
```

The code does **not** hard-code Kaggle column names. Set them in config.

## Quick start

### 1) Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Prepare folds and normalized views

```bash
bash scripts/prepare_data.sh
```

### 3) Train a baseline

```bash
bash scripts/train_baseline.sh
```

### 4) Train the hybrid model

```bash
bash scripts/train_hybrid.sh
```

### 5) Run evaluation and reports

```bash
bash scripts/eval_all.sh
```

## Recommended first run order

1. `ByT5` baseline on raw transliteration
2. baseline + metadata prefix
3. baseline + raw/normalized dual view
4. hybrid on raw
5. hybrid + metadata
6. hybrid + dual view
7. hybrid + auxiliary heads
8. ensemble the best baseline and best hybrid

## Notes

- If `mamba_ssm` is unavailable, the hybrid encoder falls back to a bi-GRU. This is deliberate so you can test the rest of the stack before wiring the exact SSM implementation you want.
- Metrics include `sacreBLEU`, `chrF`, number exact match, name span F1, damage hallucination proxy, and length ratio.
- Metadata injection is config-driven. Do not add fields until you have an ablation proving they help.

## Main docs

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/experiment_matrix.md`](docs/experiment_matrix.md)
- [`docs/data_contract.md`](docs/data_contract.md)
- [`docs/runbook.md`](docs/runbook.md)
- [`docs/metrics.md`](docs/metrics.md)
