# prepare_dataset.py
import pandas as pd
import os

# Paths
fake_path = os.path.join("dataset", "Fake.csv")
true_path = os.path.join("dataset", "True.csv")
out_path = os.path.join("dataset", "combined_news.csv")

# Read
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add label: FAKE -> 0, TRUE -> 1
fake['label'] = 0
true['label'] = 1

# Keep only necessary columns and rename
def ensure_cols(df):
    # some files have slightly different names; make robust
    cols = df.columns.str.lower()
    if 'title' in cols and 'text' in cols:
        return df[['title','text','label']]
    # fallback: try common names
    col_map = {}
    if 'headline' in cols:
        col_map['headline'] = 'title'
    if 'content' in cols:
        col_map['content'] = 'text'
    df = df.rename(columns=col_map)
    return df[['title','text','label']]

fake = ensure_cols(fake)
true = ensure_cols(true)

# Combine and shuffle
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined CSV
df.to_csv(out_path, index=False)
print(f"Saved combined dataset to {out_path} | shape: {df.shape}")
