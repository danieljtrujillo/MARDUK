"""Check for truncated transliterations by comparing train.csv with published_texts.csv."""
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
pt = pd.read_csv("data/raw/published_texts.csv")

merged = train.merge(
    pt[["oare_id", "transliteration", "transliteration_orig"]],
    on="oare_id",
    suffixes=("_train", "_pt"),
)
print(f"Merged: {len(merged)}/{len(train)} matched")

merged["train_len"] = merged["transliteration_train"].str.len()
merged["pt_len"] = merged["transliteration_pt"].fillna("").str.len()
merged["orig_len"] = merged["transliteration_orig"].fillna("").str.len()
merged["len_diff"] = merged["pt_len"] - merged["train_len"]

# Cases where published_texts has significantly longer transliteration
longer = merged[merged["len_diff"] > 10].sort_values("len_diff", ascending=False)
print(f"Cases where published_texts is >10 chars longer: {len(longer)}")
if len(longer) > 0:
    print(f"Mean difference: {longer['len_diff'].mean():.0f} chars")
    print(f"Max difference: {longer['len_diff'].max()} chars")
    print()

    for _, row in longer.head(10).iterrows():
        tid = row["oare_id"]
        tl = int(row["train_len"])
        pl = int(row["pt_len"])
        tt = str(row["transliteration_train"])[:80]
        pt_text = str(row["transliteration_pt"])[:80]
        print(f"  {tid}: train={tl}, pt={pl}, diff={pl-tl}")
        print(f"    TRAIN: {tt}...")
        print(f"    PT:    {pt_text}...")
        print()

# Also check if train starts with the same prefix (confirming truncation vs different content)
print("\n--- Prefix match analysis ---")
prefix_match = 0
for _, row in longer.iterrows():
    tt = str(row["transliteration_train"])
    pt_text = str(row["transliteration_pt"])
    if pt_text.startswith(tt[:min(30, len(tt))]):
        prefix_match += 1
print(f"Train is a prefix of PT: {prefix_match}/{len(longer)}")

# Also check: are there rows in train NOT in published_texts?
not_in_pt = set(train["oare_id"]) - set(pt["oare_id"])
print(f"\nTrain IDs not in published_texts: {len(not_in_pt)}")

# Check for test too
test = pd.read_csv("data/raw/test.csv")
test_merged = test.merge(pt[["oare_id", "transliteration"]], left_on="text_id", right_on="oare_id", suffixes=("_test", "_pt"))
print(f"\nTest merged: {len(test_merged)}/{len(test)} matched")
test_merged["test_len"] = test_merged["transliteration_test"].str.len()
test_merged["pt_len"] = test_merged["transliteration_pt"].fillna("").str.len()
test_merged["len_diff"] = test_merged["pt_len"] - test_merged["test_len"]
test_longer = test_merged[test_merged["len_diff"] > 10]
print(f"Test cases where PT is >10 chars longer: {len(test_longer)}")
