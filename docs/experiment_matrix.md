# Experiment Matrix

Run in this order.

## Baselines

1. `byt5_base` + raw
2. `byt5_base` + raw + metadata
3. `byt5_base` + raw + normalized
4. `mt5_base` + raw
5. `mt5_base` + raw + metadata
6. `mt5_base` + raw + normalized

## Hybrid

7. hybrid + raw
8. hybrid + raw + metadata
9. hybrid + raw + normalized
10. hybrid + raw + normalized + auxiliary heads
11. hybrid + source-side pretraining checkpoint init
12. hybrid + targeted oversampling
13. hybrid + self-training

## Model selection

Use:
- `sacreBLEU`
- `chrF`
- name span F1
- number exact match
- longest quartile metrics

## Stop rule

Stop hybrid work if it fails to beat the strongest baseline on at least two of:
- name span F1
- number exact match
- longest quartile BLEU/chrF
- damage hallucination proxy
