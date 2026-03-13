# Data Contract

This repo does not assume fixed Kaggle column names. Configure them in YAML.

## Required fields

### Train
- source column
- target column

### Optional
- metadata columns
- unique id column
- fold column if already prepared

## File locations

```text
data/raw/train.csv
data/raw/test.csv
```

## Config fields

```yaml
paths:
  train_csv: data/raw/train.csv
  test_csv: data/raw/test.csv

columns:
  id: text_id
  source: source_text
  target: target_text
  metadata:
    - period
    - genre
    - provenance
```

## Normalization

Normalization is intentionally conservative.

Safe operations:
- trim repeated whitespace
- normalize Unicode dashes/quotes
- separate repeated punctuation where needed
- preserve scholarly distinctions unless you have evidence they are noise

Do not:
- erase uncertainty markers
- erase damage markers
- collapse determinatives or other conventions blindly
- rewrite number strings without logging the rule
