"""Convert Mamba-ByT5 model → plain ByT5 for Kaggle T4 inference.

Run on pod (where mamba-ssm is available):
    python scripts/convert_mamba_to_plain.py

This extracts the ByT5 backbone from the Mamba-enhanced model,
unwraps the encoder, and saves as standard HF save_pretrained format.
"""
from pathlib import Path
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAMBA_CKPT = Path("outputs/runs/mamba_byt5/best")
OUTPUT_DIR = Path("outputs/runs/mamba_byt5_plain")


def main():
    print("Loading Mamba-ByT5 model...")
    from src.models.mamba_adapter_byt5 import load_mamba_byt5

    model, tokenizer = load_mamba_byt5(str(MAMBA_CKPT))
    model.eval()
    print(f"Loaded. Encoder type: {type(model.encoder).__name__}")

    # Unwrap: replace the MambaEnhancedEncoderWrapper with the original T5 encoder
    original_encoder = model.encoder.original_encoder
    model.encoder = original_encoder
    print(f"Unwrapped. Encoder type: {type(model.encoder).__name__}")

    # Save as standard HF format
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Verify it loads cleanly
    print("Verifying reload...")
    model2 = AutoModelForSeq2SeqLM.from_pretrained(str(OUTPUT_DIR))
    print(f"Reloaded OK. Parameters: {sum(p.numel() for p in model2.parameters()) / 1e6:.1f}M")

    # Quick shape check
    dummy_ids = torch.tensor([[1, 50, 100, 2]])
    with torch.no_grad():
        out = model2.generate(dummy_ids, max_new_tokens=10)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Quick test: '{decoded[:100]}'")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
