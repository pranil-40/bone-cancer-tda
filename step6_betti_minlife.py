# step6: re-extract Betti vectors at 64x64, but this time drop persistence pairs
# with lifetime below MIN_LIFE (i.e. filter out short-lived, probably-noise features).
# Run once per MIN_LIFE value (0/5/10/20) to build the files step7 compares.

import numpy as np, pandas as pd, gudhi, time, os
from PIL import Image
from multiprocessing import Pool, cpu_count

base = r"C:\projects\bone_cancer_tda"
SIZE = 64
MIN_LIFE = 20  # change this to test filtering: try 0, 5, 10, 20

def load_split(split):
    """Load one split's images (resized to SIZE x SIZE RGB) and cancer labels."""
    df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
    df.columns = df.columns.str.strip()
    imgs, labels = [], []
    for _, row in df.iterrows():
        p = os.path.join(base, split, row["filename"])
        if os.path.exists(p):
            imgs.append(np.array(Image.open(p).convert("RGB").resize((SIZE, SIZE))))
            labels.append(int(row["cancer"]))
    return np.array(imgs), np.array(labels)

def filter_life(pairs, min_life):
    """Drop persistence pairs shorter-lived than min_life; permanent (infinite) features always survive."""
    if len(pairs) == 0:
        return pairs
    life = pairs[:, 1] - pairs[:, 0]
    keep = np.isinf(pairs[:, 1]) | (life >= min_life)  # always keep infinite (permanent) features
    return pairs[keep]

def betti_vector(img, n_thresholds=100, min_life=MIN_LIFE):
    """Betti curve extraction as in step2_betti_fast.py, plus a min-lifetime filter before counting."""
    thresholds = np.linspace(0, 255, n_thresholds)
    channels = [img.mean(axis=2), img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)]
    out = []
    for ch in channels:
        cc = gudhi.CubicalComplex(dimensions=[SIZE, SIZE], top_dimensional_cells=ch.flatten())
        cc.compute_persistence()
        for dim in [0, 1]:
            pairs = filter_life(cc.persistence_intervals_in_dimension(dim), min_life)
            b = np.array([np.sum((pairs[:,0]<=t)&(pairs[:,1]>t)) for t in thresholds]) if len(pairs) else np.zeros(n_thresholds)
            out.append(b)
    return np.concatenate(out)

def process(split):
    """Run betti_vector() over an entire split in parallel across all CPU cores."""
    print(f"Loading {split}...")
    imgs, labels = load_split(split)
    print(f"Computing {len(imgs)} vectors, min_life={MIN_LIFE}, using {cpu_count()} cores...")
    start = time.time()
    with Pool(cpu_count()) as pool:
        X = np.array(pool.map(betti_vector, imgs))
    print(f"  Done in {time.time()-start:.0f}s")
    return X, labels

if __name__ == "__main__":
    X_train, y_train = process("train")
    X_val, y_val = process("valid")
    X_test, y_test = process("test")
    np.savez(f"betti_vectors_bone_minlife{MIN_LIFE}.npz", X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Saved betti_vectors_bone_minlife{MIN_LIFE}.npz")