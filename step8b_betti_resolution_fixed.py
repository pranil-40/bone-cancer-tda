# step8b: re-extracts Betti vectors at 128x128 and 224x224, same math as
# step8_betti_resolution.py, but reading from splits_fixed/ (produced by
# step0_fix_splits.py) instead of the original leaky train/valid/test folders.
# step8_betti_resolution.py is left as-is as the historical record of the
# leaky run.

import numpy as np, pandas as pd, gudhi, time, os
from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

base = r"C:\projects\bone_cancer_tda"
MIN_LIFE = 0


def load_split(split, size):
    """Load one corrected split's images (resized to size x size) from the fixed manifest."""
    df = pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))
    imgs, labels = [], []
    for _, row in df.iterrows():
        p = os.path.join(base, row["source_split"], row["filename"])
        if os.path.exists(p):
            imgs.append(np.array(Image.open(p).convert("RGB").resize((size, size))))
            labels.append(int(row["cancer"]))
    return np.array(imgs), np.array(labels)


def betti_vector(img, size, n_thresholds=100):
    """Sweep sublevel-set thresholds over each channel, same logic as step8_betti_resolution.py."""
    thresholds = np.linspace(0, 255, n_thresholds)
    channels = [img.mean(axis=2), img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)]
    out = []
    for ch in channels:
        cc = gudhi.CubicalComplex(dimensions=[size, size], top_dimensional_cells=ch.flatten())
        cc.compute_persistence()
        for dim in [0, 1]:
            pairs = cc.persistence_intervals_in_dimension(dim)
            if len(pairs):
                life = pairs[:, 1] - pairs[:, 0]
                pairs = pairs[np.isinf(pairs[:, 1]) | (life >= MIN_LIFE)]
            b = np.array([np.sum((pairs[:, 0] <= t) & (pairs[:, 1] > t)) for t in thresholds]) if len(pairs) else np.zeros(n_thresholds)
            out.append(b)
    return np.concatenate(out)


def process(split, size):
    """Load a corrected split at the given resolution and compute its Betti feature matrix in parallel."""
    print(f"{split}: loading at {size}x{size}...")
    imgs, labels = load_split(split, size)
    print(f"{split}: computing {len(imgs)} vectors at {size}x{size}...")
    start = time.time()
    with Pool(cpu_count()) as pool:
        X = np.array(pool.map(partial(betti_vector, size=size), imgs))
    print(f"{split}: done in {time.time()-start:.0f}s")
    return X, labels


if __name__ == "__main__":
    for SIZE in [128, 224]:
        X_train, y_train = process("train", SIZE)
        X_val, y_val = process("valid", SIZE)
        X_test, y_test = process("test", SIZE)
        np.savez(f"betti_vectors_bone_{SIZE}_fixed.npz", X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
        print(f"Saved betti_vectors_bone_{SIZE}_fixed.npz")
