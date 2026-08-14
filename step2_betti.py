# step2: first pass at turning X-ray images into Betti curve feature vectors.
# recomputes a fresh cubical complex per threshold, so this is slow - see
# step2_betti_fast.py for the version that fixes that.

import numpy as np
import pandas as pd
import gudhi
import time
import os
from PIL import Image

base = r"C:\projects\bone_cancer_tda"
#uhhhh i think  800-dimensional vectors - 4 channels × 100 thresholds × 2 Betti numbers = 200 dimensions per channel × 4 = 800 total.
def load_split(split):
    """Load one split's images (resized to 64x64 RGB) and cancer labels from its _classes.csv."""
    df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
    df.columns = df.columns.str.strip()
    images, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(base, split, row["filename"])
        if not os.path.exists(path):
            continue
        img = np.array(Image.open(path).convert("RGB").resize((64, 64)))
        images.append(img)
        labels.append(int(row["cancer"]))
    return np.array(images), np.array(labels)

def betti_vector(image_rgb, n_thresholds=100):
    """Sweep sublevel-set thresholds over each channel and return concatenated b0/b1 curves (800-dim)."""
    thresholds = np.linspace(0, 255, n_thresholds)
    channels = [
        image_rgb.mean(axis=2),        # grayscale
        image_rgb[:, :, 0].astype(float),  # red
        image_rgb[:, :, 1].astype(float),  # green
        image_rgb[:, :, 2].astype(float),  # blue
    ]
    result = []
    for ch in channels:
        b0, b1 = np.zeros(n_thresholds), np.zeros(n_thresholds)
        for i, t in enumerate(thresholds):
            binary = (ch <= t).astype(np.float32)
            if binary.sum() == 0 or binary.sum() == binary.size:
                continue
            cc = gudhi.CubicalComplex(dimensions=[64, 64],
                                      top_dimensional_cells=binary.flatten())
            cc.compute_persistence()
            bn = cc.betti_numbers()
            b0[i] = bn[0] if len(bn) > 0 else 0
            b1[i] = bn[1] if len(bn) > 1 else 0
        result.extend([b0, b1])
    return np.concatenate(result)

def process_split(split):
    """Run betti_vector() over an entire split and report progress/ETA as it goes."""
    print(f"Loading {split}...")
    images, labels = load_split(split)
    print(f"Computing Betti vectors for {len(images)} images...")
    start = time.time()
    X = np.zeros((len(images), 800))
    for i, img in enumerate(images):
        X[i] = betti_vector(img)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(images) - i - 1) / rate
            print(f"  {i+1}/{len(images)} — {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
    print(f"  Done in {time.time()-start:.0f}s")
    return X, labels

X_train, y_train = process_split("train")
X_val,   y_val   = process_split("valid")
X_test,  y_test  = process_split("test")

np.savez("betti_vectors_bone.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val,     y_val=y_val,
         X_test=X_test,   y_test=y_test)

print("Saved to betti_vectors_bone.npz")
print(f"Feature vector shape: {X_train.shape}")