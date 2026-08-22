# step0b: two sanity checks on the corrected splits, requested before trusting
# the step9c/step10 AUCs -
#   1. cancer vs. normal class stats (size/aspect ratio/brightness) in train,
#      checking for a non-organic shortcut a classifier could latch onto
#   2. perceptual-hash near-duplicate check between a test sample and all of
#      train, to catch visually-near-identical images (crop/flip/rotation)
#      that the exact-filename dedup in step0_fix_splits.py wouldn't catch

import os
import random

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

base = r"C:\projects\bone_cancer_tda"
random.seed(42)


def load_manifest(split):
    """Read one corrected splits_fixed/*.csv manifest."""
    return pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))


def image_path(row):
    return os.path.join(base, row["source_split"], row["filename"])


def class_stats():
    """Compare width, height, aspect ratio, and mean brightness between cancer/normal in train."""
    df = load_manifest("train")
    rows = []
    for _, r in df.iterrows():
        p = image_path(r)
        if not os.path.exists(p):
            continue
        img = Image.open(p)
        w, h = img.size
        gray = np.array(img.convert("L"), dtype=np.float64)
        rows.append({
            "cancer": r["cancer"],
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "mean_brightness": gray.mean(),
        })
    stats = pd.DataFrame(rows)

    print(f"Loaded {len(stats)} train images for class-stats check.\n")
    grouped = stats.groupby("cancer")[["width", "height", "aspect_ratio", "mean_brightness"]].agg(["mean", "std"])
    print(grouped)
    print()
    for col in ["width", "height", "aspect_ratio", "mean_brightness"]:
        cancer_mean = stats.loc[stats["cancer"] == 1, col].mean()
        normal_mean = stats.loc[stats["cancer"] == 0, col].mean()
        pooled_std = stats[col].std()
        gap = cancer_mean - normal_mean
        effect_size = gap / pooled_std if pooled_std else float("nan")
        print(f"{col}: cancer={cancer_mean:.3f}  normal={normal_mean:.3f}  "
              f"gap={gap:+.3f}  effect_size(Cohen's d)={effect_size:+.3f}")
    return stats


def near_duplicate_check(n_test_samples=30, hash_size=16, threshold=10):
    """Sample test images, phash them, and compare against phashes of every train image
    to catch visually-near-identical pairs the filename dedup missed."""
    train_df = load_manifest("train")
    test_df = load_manifest("test")

    print(f"\nHashing {len(train_df)} train images (phash, hash_size={hash_size})...")
    train_hashes = []
    for _, r in train_df.iterrows():
        p = image_path(r)
        if not os.path.exists(p):
            continue
        h = imagehash.phash(Image.open(p), hash_size=hash_size)
        train_hashes.append((r["filename"], h))
    print(f"  hashed {len(train_hashes)} train images")

    sample = test_df.sample(n=min(n_test_samples, len(test_df)), random_state=42)
    print(f"\nChecking {len(sample)} random test images against train (Hamming distance <= {threshold} = near-duplicate)...")

    results = []
    for _, r in sample.iterrows():
        p = image_path(r)
        if not os.path.exists(p):
            continue
        test_hash = imagehash.phash(Image.open(p), hash_size=hash_size)
        best_name, best_dist = None, None
        for train_name, train_hash in train_hashes:
            d = test_hash - train_hash
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, train_name
        is_dup = best_dist <= threshold
        results.append({"test_file": r["filename"], "closest_train_file": best_name,
                         "hamming_distance": best_dist, "near_duplicate": is_dup})
        flag = "NEAR-DUP" if is_dup else ""
        print(f"  {r['filename'][:50]:50s} closest={best_name[:50]:50s} dist={best_dist:3d} {flag}")

    results_df = pd.DataFrame(results)
    n_dup = results_df["near_duplicate"].sum()
    print(f"\n{n_dup}/{len(results_df)} sampled test images ({n_dup/len(results_df):.1%}) "
          f"have a near-duplicate (hamming <= {threshold}) somewhere in train.")
    return results_df


if __name__ == "__main__":
    print("=== Check 1: class-level stats (cancer vs normal, train) ===\n")
    class_stats()
    print("\n=== Check 2: perceptual-hash near-duplicate check (test sample vs train) ===")
    near_duplicate_check()
