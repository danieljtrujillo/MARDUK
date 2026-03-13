# Metrics

## Topline
- sacreBLEU
- chrF

## Structured
- number exact match
- normalized number exact match
- name span precision / recall / F1
- length ratio

## Diagnostics
- damage hallucination proxy
- copy ratio
- repeated token rate
- unknown-token rate for target tokenizer

## Slice reports
Evaluate by:
- shortest quartile
- longest quartile
- examples containing numbers
- examples containing damage markers
- metadata slices such as genre or provenance
