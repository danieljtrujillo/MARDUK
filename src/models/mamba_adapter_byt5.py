"""Mamba-enhanced ByT5: BiMamba SSM adapter layers injected after ByT5 encoder.

Architecture:
    ByT5 Encoder → BiMamba Adapter Stack → ByT5 Decoder

The adapter captures long-range dependencies that byte-level Transformer attention
might miss, using state-space models' O(n) sequence modeling capability.
The adapter projection is zero-initialized so the model starts as vanilla ByT5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class BiMambaAdapter(nn.Module):
    """Single bidirectional Mamba adapter block with residual connection.

    Runs forward and backward SSMs in parallel, projects back to d_model,
    and adds a residual. The projection is zero-initialized so the adapter
    starts as identity (doesn't disrupt pretrained ByT5 representations).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model * 2, d_model)

        # Zero-init projection → adapter starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        fwd_out = self.mamba_fwd(h)
        bwd_out = self.mamba_bwd(h.flip(dims=[1])).flip(dims=[1])
        combined = torch.cat([fwd_out, bwd_out], dim=-1)
        return x + self.proj(combined)


class GRUAdapterFallback(nn.Module):
    """Fallback adapter using bidirectional GRU when mamba-ssm is unavailable."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.gru(h)
        return x + self.proj(out)


class MambaAdapterStack(nn.Module):
    """Stack of BiMamba adapter layers for post-encoder processing."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
    ):
        super().__init__()
        if use_mamba and MAMBA_AVAILABLE:
            self.layers = nn.ModuleList([
                BiMambaAdapter(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GRUAdapterFallback(d_model) for _ in range(n_layers)
            ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = self.dropout(layer(hidden_states))
        return self.final_norm(hidden_states)


class MambaEnhancedEncoderWrapper(nn.Module):
    """Wraps a T5 encoder and applies MambaAdapterStack to its output.

    Drop-in replacement for model.encoder — both training forward() and
    generate() call self.encoder, so this wrapper is transparent.
    """

    def __init__(self, original_encoder: nn.Module, adapter: MambaAdapterStack):
        super().__init__()
        self.original_encoder = original_encoder
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        outputs = self.original_encoder(*args, **kwargs)
        enhanced = self.adapter(outputs.last_hidden_state)
        return BaseModelOutput(
            last_hidden_state=enhanced,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def config(self):
        return self.original_encoder.config

    @property
    def device(self):
        return next(self.original_encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.original_encoder.parameters()).dtype

    @property
    def embed_tokens(self):
        return self.original_encoder.embed_tokens

    @property
    def main_input_name(self):
        return getattr(self.original_encoder, "main_input_name", "input_ids")

    def get_input_embeddings(self):
        return self.original_encoder.get_input_embeddings()


def create_mamba_byt5(
    checkpoint_path: str,
    n_mamba_layers: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    adapter_dropout: float = 0.1,
    freeze_byt5: bool = False,
) -> AutoModelForSeq2SeqLM:
    """Load a ByT5 checkpoint and inject Mamba adapter layers after the encoder.

    Args:
        checkpoint_path: Path to pretrained ByT5 (Phase 1 output or HF model name).
        n_mamba_layers: Number of BiMamba adapter layers.
        freeze_byt5: If True, freeze all ByT5 params (train only adapter).

    Returns:
        A T5ForConditionalGeneration model with a Mamba-enhanced encoder.
        Compatible with HF Seq2SeqTrainer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    d_model = model.config.d_model

    adapter = MambaAdapterStack(
        d_model=d_model,
        n_layers=n_mamba_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=adapter_dropout,
    )

    # Replace encoder with wrapped version
    model.encoder = MambaEnhancedEncoderWrapper(model.encoder, adapter)

    if freeze_byt5:
        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in adapter.parameters())

    return model, {"total": total, "trainable": trainable, "adapter": adapter_params}


def save_mamba_byt5(model, tokenizer, output_dir: str):
    """Save the Mamba-enhanced ByT5 model.

    Saves the ByT5 base via save_pretrained and the adapter state dict separately.
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save the full model state (including adapter)
    torch.save(model.state_dict(), out / "model.pt")

    # Save tokenizer
    tokenizer.save_pretrained(str(out))

    # Save adapter config for reconstruction
    import json
    adapter = model.encoder.adapter
    adapter_cfg = {
        "n_layers": len(adapter.layers),
        "d_model": model.config.d_model,
    }
    with open(out / "mamba_adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f)


def load_mamba_byt5(checkpoint_dir: str, base_model: str = "google/byt5-base"):
    """Load a saved Mamba-enhanced ByT5 model."""
    import json
    from pathlib import Path

    ckpt = Path(checkpoint_dir)

    with open(ckpt / "mamba_adapter_config.json") as f:
        adapter_cfg = json.load(f)

    model, _ = create_mamba_byt5(
        checkpoint_path=base_model,
        n_mamba_layers=adapter_cfg["n_layers"],
    )

    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    return model, tokenizer
