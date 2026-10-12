# same as betti_minlife.py but varying image size instead. run once with
# SIZE=128 and once with 224. reads the original folders so this is the leaky
# version, betti_size_dedup.py uses the deduped splits

import numpy as np, pandas as pd, gudhi, time, os
from PIL import Image
from multiprocessing import Pool, cpu_count

base = r"C:\projects\bone_cancer_tda"
SIZE = 224       # change to 128 or 224 for each run
MIN_LIFE = 0     # best value from compare_minlife.py

def load_split(split):
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
    np.savez(f"betti_{SIZE}_orig.npz", X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Saved betti_{SIZE}_orig.npz")