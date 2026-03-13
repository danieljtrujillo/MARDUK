from __future__ import annotations

import re
from typing import Iterable

import evaluate
import numpy as np


_bleu = evaluate.load("sacrebleu")
_chrf = evaluate.load("chrf")


def compute_generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    bleu = _bleu.compute(predictions=predictions, references=[[x] for x in references])
    chrf = _chrf.compute(predictions=predictions, references=references)
    return {
        "sacrebleu": float(bleu["score"]),
        "chrf": float(chrf["score"]),
        "avg_pred_len": float(np.mean([len(x.split()) for x in predictions])) if predictions else 0.0,
        "avg_ref_len": float(np.mean([len(x.split()) for x in references])) if references else 0.0,
    }


_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def extract_numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text or "")


def number_exact_match(predictions: list[str], references: list[str]) -> float:
    matches = 0
    total = max(1, len(predictions))
    for pred, ref in zip(predictions, references):
        matches += int(extract_numbers(pred) == extract_numbers(ref))
    return matches / total


_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z\-']+\b")


def extract_name_spans(text: str) -> set[str]:
    return set(_NAME_RE.findall(text or ""))


def name_span_f1(predictions: list[str], references: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        p = extract_name_spans(pred)
        r = extract_name_spans(ref)
        tp += len(p & r)
        fp += len(p - r)
        fn += len(r - p)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"name_precision": precision, "name_recall": recall, "name_f1": f1}


def length_ratio(predictions: list[str], references: list[str]) -> float:
    pred_len = sum(max(1, len(p.split())) for p in predictions)
    ref_len = sum(max(1, len(r.split())) for r in references)
    return pred_len / max(1, ref_len)


def damage_hallucination_proxy(predictions: list[str], references: list[str]) -> float:
    """
    Cheap proxy:
    when ref contains uncertainty markers such as [...] or x and prediction becomes fully fluent,
    count it as a possible hallucination. Replace with domain-specific rules later.
    """
    hits = 0
    total = 0
    for pred, ref in zip(predictions, references):
        if any(tok in ref for tok in ["[", "]", " x ", "?"]):
            total += 1
            pred_numbers = len(extract_numbers(pred))
            ref_numbers = len(extract_numbers(ref))
            if abs(pred_numbers - ref_numbers) > 1:
                hits += 1
    return hits / max(1, total)


def all_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    out = compute_generation_metrics(predictions, references)
    out["number_exact_match"] = number_exact_match(predictions, references)
    out["length_ratio"] = length_ratio(predictions, references)
    out["damage_hallucination_proxy"] = damage_hallucination_proxy(predictions, references)
    out.update(name_span_f1(predictions, references))
    # Competition metric: geometric mean of BLEU and chrF++
    bleu_val = max(out["sacrebleu"], 0.0)
    chrf_val = max(out["chrf"], 0.0)
    out["competition_score"] = float(np.sqrt(bleu_val * chrf_val))
    return out
