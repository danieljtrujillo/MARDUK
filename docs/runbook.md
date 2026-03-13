# Runbook

## Prepare

1. Put CSV files in `data/raw/`
2. Edit `configs/data/raw.yaml` with the actual column names
3. Run:

```bash
bash scripts/prepare_data.sh
```

This writes:
- normalized CSVs in `data/processed/`
- fold assignments in `data/processed/folds.csv`

## Baseline

```bash
bash scripts/train_baseline.sh
```

Outputs:
- checkpoints in `outputs/runs/<run_name>/`
- validation predictions
- metrics JSON
- error-bucket CSV

## Hybrid

```bash
bash scripts/train_hybrid.sh
```

Outputs:
- checkpoints
- validation predictions
- metrics JSON
- auxiliary-head metrics

## Evaluation

```bash
bash scripts/eval_all.sh
```

This aggregates:
- topline metrics
- slice metrics
- error buckets
- candidate ensemble inputs

## Common failure checks

### Wrong numbers
Check:
- tokenization
- target truncation
- decoding constraints

### Dropped names
Check:
- metadata leakage from the target
- name BIO head labels
- copy-bias settings

### Long input collapse
Check:
- source max length
- batch packing
- curriculum buckets
- encoder hidden-state memory use

### Damage hallucination
Check:
- uncertainty tag labels
- decoding length penalty
- oversampling of damaged examples
