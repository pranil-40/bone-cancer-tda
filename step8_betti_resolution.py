# step8 (renamed from step8_betti_multi.py): same idea as step6, but the axis
# being swept is image resolution (SIZE) instead of MIN_LIFE - run once each for
# SIZE=128 and SIZE=224 to build the files step9 compares against the 64x64 baseline.

import numpy as np, pandas as pd, gudhi, time, os
from PIL import Image
from multiprocessing import Pool, cpu_count

base = r"C:\projects\bone_cancer_tda"
SIZE = 224       # change to 128 or 224 for each run
MIN_LIFE = 0     # carried over from step7's winner

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

def betti_vector(img, n_thresholds=100):
    """Betti curve extraction at whatever SIZE is set above, with the step6 min-life filter applied."""
    thresholds = np.linspace(0, 255, n_thresholds)
    channels = [img.mean(axis=2), img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)]
    out = []
    for ch in channels:
        cc = gudhi.CubicalComplex(dimensions=[SIZE, SIZE], top_dimensional_cells=ch.flatten())
        cc.compute_persistence()
        for dim in [0, 1]:
            pairs = cc.persistence_intervals_in_dimension(dim)
            if len(pairs):
                life = pairs[:, 1] - pairs[:, 0]
                pairs = pairs[np.isinf(pairs[:, 1]) | (life >= MIN_LIFE)]
            b = np.array([np.sum((pairs[:,0]<=t)&(pairs[:,1]>t)) for t in thresholds]) if len(pairs) else np.zeros(n_thresholds)
            out.append(b)
    return np.concatenate(out)

def process(split):
    """Run betti_vector() over an entire split in parallel across all CPU cores."""
    print(f"{split}: loading...")
    imgs, labels = load_split(split)
    print(f"{split}: computing {len(imgs)} vectors at {SIZE}x{SIZE}...")
    start = time.time()
    with Pool(cpu_count()) as pool:
        X = np.array(pool.map(betti_vector, imgs))
    print(f"{split}: done in {time.time()-start:.0f}s")
    return X, labels

if __name__ == "__main__":
    X_train, y_train = process("train")
    X_val, y_val = process("valid")
    X_test, y_test = process("test")
    np.savez(f"betti_vectors_bone_{SIZE}.npz", X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Saved betti_vectors_bone_{SIZE}.npz")