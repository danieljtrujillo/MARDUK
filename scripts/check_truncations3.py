"""Analyze and fix truncated transliterations in training data.

Strategy: Replace truncated transliterations in train.csv with full versions from published_texts.csv.
The translations in train.csv are already full-length, so only the source needs fixing.
"""
import pandas as pd
import numpy as np

train = pd.read_csv("data/raw/train.csv")
pt = pd.read_csv("data/raw/published_texts.csv")

# Merge to get full transliterations
merged = train.merge(
    pt[["oare_id", "transliteration"]],
    on="oare_id",
    suffixes=("_train", "_pt"),
)

# Detect truncation: PT version is at least 10 chars longer AND train is a prefix of PT
truncated_mask = pd.Series(False, index=merged.index)
for idx, row in merged.iterrows():
    t = str(row["transliteration_train"])
    p = str(row["transliteration_pt"])
    if len(p) > len(t) + 10:
        # Check prefix match (first 30 chars)
        if p[:min(30, len(t))].lower() == t[:min(30, len(t))].lower():
            truncated_mask[idx] = True

truncated = merged[truncated_mask]
print(f"Truncated transliterations to fix: {len(truncated)}/{len(train)} ({100*len(truncated)/len(train):.1f}%)")
print(f"Average chars recovered: {(merged.loc[truncated_mask, 'transliteration_pt'].str.len() - merged.loc[truncated_mask, 'transliteration_train'].str.len()).mean():.0f}")
print()

# Show length distributions before and after
print("Before fix:")
print(f"  Mean source length: {train['transliteration'].str.len().mean():.0f}")
print(f"  Max source length:  {train['transliteration'].str.len().max()}")

# Apply fix: replace truncated transliterations with full versions
fixed = train.copy()
for idx, row in truncated.iterrows():
    oare_id = row["oare_id"]
    full_translit = row["transliteration_pt"]
    mask = fixed["oare_id"] == oare_id
    fixed.loc[mask, "transliteration"] = full_translit

print("\nAfter fix:")
print(f"  Mean source length: {fixed['transliteration'].str.len().mean():.0f}")
print(f"  Max source length:  {fixed['transliteration'].str.len().max()}")

# Check byte-level impact (ByT5 uses bytes, not chars)
fixed_bytes = fixed["transliteration"].apply(lambda x: len(str(x).encode("utf-8")))
print(f"\nByte-level stats (ByT5-relevant):")
print(f"  Mean bytes: {fixed_bytes.mean():.0f}")
print(f"  Max bytes:  {fixed_bytes.max()}")
print(f"  >1024 bytes: {(fixed_bytes > 1024).sum()}")
print(f"  >2048 bytes: {(fixed_bytes > 2048).sum()}")
print(f"  >4096 bytes: {(fixed_bytes > 4096).sum()}")

# Show distribution of the recovered lengths
print(f"\nRecovered transliteration byte lengths (for the 250 fixed rows):")
fixed_ids = set(truncated["oare_id"])
fixed_subset = fixed[fixed["oare_id"].isin(fixed_ids)]
byte_lens = fixed_subset["transliteration"].apply(lambda x: len(str(x).encode("utf-8")))
for threshold in [1024, 1500, 2000, 3000, 4000, 5000]:
    count = (byte_lens > threshold).sum()
    print(f"  >{threshold} bytes: {count}")
