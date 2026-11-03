"""Step 1: load ISEAR, clean it, print basic statistics, save data/isear_clean.csv."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

LABELS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
PLACEHOLDERS = {"nothing", "none", "blank", "not applicable", "no response"}

df = load_dataset("gsri-18/ISEAR-dataset-complete", split="train").to_pandas()
print(f"Raw rows: {len(df)}")

# 3 texts contain a comma that the CSV split into an extra column: glue them back.
extra = df["Unnamed: 2"].notna()
df.loc[extra, "content"] = df.loc[extra, "content"] + "," + df.loc[extra, "Unnamed: 2"]
df = df.rename(columns={"content": "text", "emotion": "label"})[["text", "label"]]
df["text"] = df["text"].str.strip()

# Normalised text (lower case, no punctuation) is only used to find placeholders and duplicates.
norm = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()

placeholder = norm.isin(PLACEHOLDERS)                                     # e.g. "NO RESPONSE."
conflict = (df.groupby(norm)["label"].transform("nunique") > 1) & ~placeholder  # same text, different labels
duplicate = norm.duplicated() & ~conflict & ~placeholder                         # same text, same label: keep first

df = df[~(placeholder | conflict | duplicate)].reset_index(drop=True)
print(f"Removed: {placeholder.sum()} placeholders, {conflict.sum()} conflicting labels, "
      f"{duplicate.sum()} duplicates -> {len(df)} rows left")

df["label_id"] = df["label"].map(LABELS.index)
print("\nExamples per class:")
print(df["label"].value_counts().reindex(LABELS).to_string())
print(f"\nMedian length: {df['text'].str.split().str.len().median():.0f} words")

Path("data").mkdir(exist_ok=True)
df.to_csv("data/isear_clean.csv", index=False)
print("Saved data/isear_clean.csv")
