from __future__ import annotations

import torch
from torch import nn


class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)
