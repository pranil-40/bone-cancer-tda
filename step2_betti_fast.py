# step2 (fast version): same Betti-vector extraction as step2_betti.py, but builds
# one cubical complex per channel instead of one per threshold, and parallelizes
# across images with multiprocessing. Output should match step2_betti.py, just faster.

import numpy as np
import pandas as pd
import gudhi
import time
import os
from PIL import Image
from multiprocessing import Pool, cpu_count

base = r"C:\projects\bone_cancer_tda"

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

def betti_vector_fast(image_rgb, n_thresholds=100):
    """Same output as betti_vector() in step2_betti.py, but reads persistence pairs once per channel
    and counts them across thresholds instead of rebuilding the complex each time."""
    thresholds = np.linspace(0, 255, n_thresholds)
    channels = [
        image_rgb.mean(axis=2),
        image_rgb[:, :, 0].astype(float),
        image_rgb[:, :, 1].astype(float),
        image_rgb[:, :, 2].astype(float),
    ]
    result = []
    for ch in channels:
        # build ONE cubical complex using pixel values directly as filtration
        cc = gudhi.CubicalComplex(dimensions=[64, 64],
                                  top_dimensional_cells=ch.flatten())
        cc.compute_persistence()
        pairs0 = cc.persistence_intervals_in_dimension(0)
        pairs1 = cc.persistence_intervals_in_dimension(1)

        b0 = np.array([np.sum((pairs0[:, 0] <= t) & (pairs0[:, 1] > t))
                       for t in thresholds]) if len(pairs0) else np.zeros(n_thresholds)
        b1 = np.array([np.sum((pairs1[:, 0] <= t) & (pairs1[:, 1] > t))
                       for t in thresholds]) if len(pairs1) else np.zeros(n_thresholds)
        result.extend([b0, b1])
    return np.concatenate(result)

def process_split(split):
    """Run betti_vector_fast() over an entire split in parallel across all CPU cores."""
    print(f"Loading {split}...")
    images, labels = load_split(split)
    print(f"Computing Betti vectors for {len(images)} images using {cpu_count()} cores...")
    start = time.time()
    with Pool(cpu_count()) as pool:
        X = np.array(pool.map(betti_vector_fast, images))
    print(f"  Done in {time.time()-start:.0f}s")
    return X, labels

if __name__ == "__main__":
    X_train, y_train = process_split("train")
    X_val,   y_val   = process_split("valid")
    X_test,  y_test  = process_split("test")

    np.savez("betti_vectors_bone.npz",
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val,
             X_test=X_test,   y_test=y_test)
    print("Saved.")