# step1: quick look at the dataset before doing anything with it -
# class counts per split, and a sanity check that images actually load and resize.

import pandas as pd
import os
from PIL import Image
import numpy as np

base = r"C:\projects\bone_cancer_tda"

for split in ["train", "valid", "test"]:
    df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
    df.columns = df.columns.str.strip()
    cancerous = df["cancer"].sum()
    normal = df["normal"].sum()
    print(f"{split}: {len(df)} images — {cancerous} cancerous, {normal} normal")

# peeking at one image
df_train = pd.read_csv(os.path.join(base, "train", "_classes.csv"))
df_train.columns = df_train.columns.str.strip()
sample_file = df_train["filename"].iloc[0]
img = Image.open(os.path.join(base, "train", sample_file)).convert("RGB").resize((64, 64))
print(f"\nSample image loaded: {sample_file}")
print(f"Shape after resize: {np.array(img).shape}")