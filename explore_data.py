"""Step 1: load ISEAR, clean it, print basic statistics, save data/isear_clean.csv."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

LABELS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

# Non-answers, matched against the normalised text (lower case, no punctuation, single spaces).
# Refusals and references to another answer: removed at any length.
NON_ANSWER = [
    r"^(?:nothing|none|blank)$",
    r"no response", r"not applicable", r"does not apply", r"not included on questionnaire",
    r"no description", r"no answer",
    r"same as in", r"as for disgust", r"mentioned under", r"see answer for",
    r"not felt this emotion",
]
# "Can't recall": only a non-answer in short texts; in long texts it introduces a real answer.
CANT_RECALL = [
    r"cant think of", r"can not think of",
    r"cant remember", r"can not remember", r"do not remember", r"unable to remember",
]
MAX_WORDS_CANT_RECALL = 20  # ~ median text length (19 words); checked by inspecting all matches

df = load_dataset("gsri-18/ISEAR-dataset-complete", split="train").to_pandas()
print(f"Raw rows: {len(df)}")

# 3 texts contain a comma that the CSV split into an extra column: glue them back.
extra = df["Unnamed: 2"].notna()
df.loc[extra, "content"] = df.loc[extra, "content"] + "," + df.loc[extra, "Unnamed: 2"]
df = df.rename(columns={"content": "text", "emotion": "label"})[["text", "label"]]

# Collapse line breaks and repeated spaces left over from the original fixed-width files.
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Normalised text is only used to find non-answers and duplicates.
norm = (df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip())

short = norm.str.split().str.len() <= MAX_WORDS_CANT_RECALL
placeholder = (norm.str.contains("|".join(NON_ANSWER), regex=True)
               | (short & norm.str.contains("|".join(CANT_RECALL), regex=True)))
conflict = (df.groupby(norm)["label"].transform("nunique") > 1) & ~placeholder  # same text, different labels
duplicate = norm.duplicated() & ~conflict & ~placeholder                         # same text, same label: keep first

# Verify every non-answer match by eye (remove this print once satisfied).
print("\nNon-answers found:")
print(df.loc[placeholder, ["text", "label"]].to_string())

df = df[~(placeholder | conflict | duplicate)].reset_index(drop=True)
print(f"\nRemoved: {placeholder.sum()} non-answers, {conflict.sum()} conflicting labels, "
      f"{duplicate.sum()} duplicates -> {len(df)} rows left")

df["label_id"] = df["label"].map(LABELS.index)
print("\nExamples per class:")
print(df["label"].value_counts().reindex(LABELS).to_string())
print(f"\nMajority baseline: {df['label'].value_counts().max() / len(df):.3f}")
print(f"Median length: {df['text'].str.split().str.len().median():.0f} words")

Path("data").mkdir(exist_ok=True)
df.to_csv("data/isear_clean.csv", index=False)
print("Saved data/isear_clean.csv")
