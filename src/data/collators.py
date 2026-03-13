from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class HybridBatch:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    labels: torch.Tensor
    aux: dict[str, torch.Tensor]


class ByteSourceEncoder:
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, max_length: int = 1024):
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8", errors="replace")[: self.max_length - 2]
        ids = [self.BOS] + [b + 3 for b in raw] + [self.EOS]
        return ids

    def pad_batch(self, batch_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch_ids)
        padded = []
        mask = []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.PAD] * pad_len)
            mask.append([1] * len(ids) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


class HybridCollator:
    def __init__(
        self,
        source_encoder: ByteSourceEncoder,
        target_tokenizer: PreTrainedTokenizerBase,
        target_max_length: int = 256,
    ) -> None:
        self.source_encoder = source_encoder
        self.target_tokenizer = target_tokenizer
        self.target_max_length = target_max_length

    def __call__(self, features: list[dict[str, Any]]) -> HybridBatch:
        source_ids = [self.source_encoder.encode(f["source"]) for f in features]
        source_ids_t, source_mask_t = self.source_encoder.pad_batch(source_ids)

        target_texts = [f["target"] for f in features]
        tok = self.target_tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=self.target_max_length,
            return_tensors="pt",
        )
        labels = tok["input_ids"].clone()
        labels[labels == self.target_tokenizer.pad_token_id] = -100

        aux_keys = ["name_labels", "number_labels", "damage_labels"]
        aux: dict[str, torch.Tensor] = {}
        src_len = source_ids_t.size(1)
        for key in aux_keys:
            seqs = []
            for f in features:
                vals = list(f.get(key, []))[:src_len]
                vals = vals + [0] * (src_len - len(vals))
                seqs.append(vals)
            aux[key] = torch.tensor(seqs, dtype=torch.long)

        return HybridBatch(
            source_ids=source_ids_t,
            source_mask=source_mask_t,
            target_ids=tok["input_ids"],
            target_mask=tok["attention_mask"],
            labels=labels,
            aux=aux,
        )
