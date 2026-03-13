from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class BiGRUFallbackEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: runs forward + backward SSM, concatenates, projects back to d_model."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        fwd_out = self.fwd(normed)
        # Reverse, run backward Mamba, reverse back
        bwd_out = self.bwd(normed.flip(dims=[1])).flip(dims=[1])
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        return x + self.proj(combined)  # residual connection


class MambaEncoderWrapper(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        dropout: float = 0.1,
        use_mamba_if_available: bool = True,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_mamba = False

        if use_mamba_if_available:
            try:
                if bidirectional:
                    blocks = nn.ModuleList(
                        [BiMambaBlock(d_model=d_model) for _ in range(n_layers)]
                    )
                else:
                    from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
                    blocks = nn.ModuleList()
                    for _ in range(n_layers):
                        blocks.append(
                            nn.Sequential(
                                nn.LayerNorm(d_model),
                                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                            )
                        )
                self.layers = blocks
                self.use_mamba = True
                self.bidirectional = bidirectional
            except Exception:
                self.layers = None
                self.bidirectional = False

        if not self.use_mamba:
            self.fallback = BiGRUFallbackEncoder(d_model=d_model, n_layers=n_layers, dropout=dropout)
            self.bidirectional = False

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        if self.use_mamba:
            if self.bidirectional:
                for layer in self.layers:
                    x = layer(x)  # residual is inside BiMambaBlock
            else:
                for layer in self.layers:
                    residual = x
                    x = layer[0](x)
                    x = layer[1](x) + residual
        else:
            x = self.fallback(x)
        return self.final_norm(x)
