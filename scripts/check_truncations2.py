"""Check AICC translations and relationship for truncated rows."""
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
pt = pd.read_csv("data/raw/published_texts.csv")

merged = train.merge(
    pt[["oare_id", "transliteration", "AICC_translation"]],
    on="oare_id",
    suffixes=("_train", "_pt"),
)
merged["train_len"] = merged["transliteration_train"].str.len()
merged["pt_len"] = merged["transliteration_pt"].fillna("").str.len()
merged["len_diff"] = merged["pt_len"] - merged["train_len"]

truncated = merged[merged["len_diff"] > 10]
has_aicc = truncated["AICC_translation"].notna().sum()
print(f"Truncated rows: {len(truncated)}")
print(f"  With AICC_translation: {has_aicc}")
print(f"  Without: {len(truncated) - has_aicc}")
print()

# Check the FULL set — do all train rows have translation?
print(f"All train rows: {len(train)}")
print(f"  With translation: {train['translation'].notna().sum()}")
print(f"  AICC translations in pt: {pt['AICC_translation'].notna().sum()}")
print()

# For truncated rows: does train.translation correspond to the truncated transliteration or the full?
# i.e., is the translation also partial?
print("--- Truncated examples (showing translation + transliteration lengths) ---")
truncated_sorted = truncated.sort_values("len_diff", ascending=False)
for _, row in truncated_sorted.head(5).iterrows():
    tid = row["oare_id"]
    t_trans = str(row["translation"])
    t_translit = str(row["transliteration_train"])
    pt_translit = str(row["transliteration_pt"])
    aicc = str(row["AICC_translation"])
    
    print(f"ID: {tid}")
    print(f"  Train translit: {len(t_translit)} chars | PT translit: {len(pt_translit)} chars (diff={len(pt_translit)-len(t_translit)})")
    print(f"  Train translation ({len(t_trans)} chars): {t_trans[:120]}...")
    if aicc != "nan":
        print(f"  AICC translation  ({len(aicc)} chars): {aicc[:120]}...")
    print()

# How many train.csv transliterations are at exactly the same char boundary?
# This could indicate a max-length cutoff
from collections import Counter
train_lens = train["transliteration"].str.len()
print(f"\nTrain transliteration length stats:")
print(f"  Max: {train_lens.max()}")
print(f"  Mean: {train_lens.mean():.0f}")
print(f"  Median: {train_lens.median():.0f}")

# Check if many cluster at the max length
max_len = train_lens.max()
at_max = (train_lens >= max_len - 5).sum()
print(f"  Rows within 5 of max ({max_len}): {at_max}")

# Length distribution near the top
bins = [800, 850, 900, 935, 940, 950, 1000, train_lens.max() + 1]
print(f"\nLength distribution (top end):")
for i in range(len(bins) - 1):
    count = ((train_lens >= bins[i]) & (train_lens < bins[i+1])).sum()
    if count > 0:
        print(f"  [{bins[i]}-{bins[i+1]}): {count}")
