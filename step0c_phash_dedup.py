# step0c: supersedes step0_fix_splits.py's filename-only dedup. The sanity
# check in step0b_sanity_checks.py showed the same source X-ray sometimes
# reappears under a completely unrelated filename (different naming
# convention entirely, e.g. a Roboflow project merge), so exact-basename
# matching missed real duplicates. This groups all 8,811 original images by
# perceptual hash (phash, Hamming distance <= 10 = same threshold used in
# the sanity check) instead, across ALL THREE original splits at once, then
# reassigns whole groups to a single split with the same proportional/
# class-balance logic as step0. Also reports mean brightness and cancer
# ratio per filename "naming family", to check whether the brightness gap
# found in step0b tracks a sub-source rather than the pathology label.
#
# Overwrites splits_fixed/*.csv - this is now the canonical corrected split.

import os
import re

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

base = r"C:\projects\bone_cancer_tda"
SPLITS = ["train", "valid", "test"]
HASH_SIZE = 16
DUP_THRESHOLD = 10
TARGET_PROPORTIONS = {"train": 7057 / 8811, "valid": 882 / 8811, "test": 872 / 8811}

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def load_all_rows():
    """Read all three original _classes.csv files into one frame, tagging each row with source_split."""
    frames = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
        df.columns = df.columns.str.strip()
        df["source_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def naming_family(filename):
    """Best-effort bucket of the filename's naming convention, stripping the roboflow
    export suffix/extension and any trailing digits, e.g. 'IMG0000345_jpg...' -> 'img'."""
    stem = re.sub(r"\.rf\.[0-9a-f]{16,}\.(jpg|jpeg|png)$", "", filename.strip(), flags=re.I)
    stem = re.sub(r"_(png|jpg|jpeg)$", "", stem, flags=re.I)
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*)", stem)
    if not m:
        return "(numeric-prefix)"
    fam = m.group(1).strip("_-").lower()
    return fam if fam else "(numeric-prefix)"


def hash_and_brightness(df):
    """Single pass over every image: compute its phash (for dedup) and mean
    grayscale brightness (for the family/brightness report) without loading twice."""
    hashes = np.zeros((len(df), HASH_SIZE * HASH_SIZE), dtype=bool)
    brightness = np.zeros(len(df), dtype=np.float64)
    missing = []
    for i, row in enumerate(df.itertuples()):
        p = os.path.join(base, row.source_split, row.filename)
        if not os.path.exists(p):
            missing.append(i)
            continue
        img = Image.open(p)
        hashes[i] = imagehash.phash(img, hash_size=HASH_SIZE).hash.flatten()
        brightness[i] = np.array(img.convert("L"), dtype=np.float64).mean()
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i+1}/{len(df)} images...")
    return hashes, brightness, missing


class UnionFind:
    """Standard disjoint-set structure for grouping images connected by a near-duplicate edge."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_dup_groups(hashes, threshold=DUP_THRESHOLD, chunk=300):
    """Pairwise Hamming distance in chunks (packed bits + popcount lookup, since a full
    NxN distance matrix would be too large), unioning any pair within threshold."""
    n = len(hashes)
    packed = np.packbits(hashes, axis=1)  # (n, 32) uint8
    uf = UnionFind(n)
    n_edges = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xor = packed[start:end, None, :] ^ packed[None, :, :]   # (c, n, 32)
        dist = POPCOUNT[xor].sum(axis=2)                        # (c, n)
        rows, cols = np.where(dist <= threshold)
        for r, c in zip(rows, cols):
            global_r = start + r
            if c > global_r:  # each pair counted once, from the lower index's chunk
                uf.union(global_r, c)
                n_edges += 1
        print(f"  compared {end}/{n} images against the full set...")
    print(f"  {n_edges} near-duplicate edges found (threshold={threshold})")
    return [uf.find(i) for i in range(n)]


def assign_groups(df):
    """Same greedy proportional/class-balance assignment as step0_fix_splits.py,
    but keyed by phash group id instead of exact base filename."""
    total_images = len(df)
    total_cancer = df["cancer"].sum()
    overall_cancer_rate = total_cancer / total_images

    groups = df.groupby("group_id")
    ordered = sorted(groups, key=lambda kv: (-len(kv[1]), kv[0]))

    target_count = {s: TARGET_PROPORTIONS[s] * total_images for s in SPLITS}
    target_cancer = {s: target_count[s] * overall_cancer_rate for s in SPLITS}

    counts = {s: 0 for s in SPLITS}
    cancer_counts = {s: 0 for s in SPLITS}
    inconsistent_groups = []

    assignment = {}
    for group_id, rows in ordered:
        labels = set(rows["cancer"])
        if len(labels) > 1:
            inconsistent_groups.append(group_id)
        gsize = len(rows)
        gcancer = rows["cancer"].sum()

        best_split, best_score = None, float("-inf")
        for split in SPLITS:
            size_deficit = (target_count[split] - counts[split]) / target_count[split]
            cancer_deficit = (target_cancer[split] - cancer_counts[split]) / max(target_cancer[split], 1)
            score = size_deficit + cancer_deficit
            if score > best_score:
                best_split, best_score = split, score

        counts[best_split] += gsize
        cancer_counts[best_split] += gcancer
        assignment[group_id] = best_split

    df["new_split"] = df["group_id"].map(assignment)
    return df, counts, cancer_counts, inconsistent_groups, overall_cancer_rate


def report_family_brightness(df):
    """Mean brightness and cancer ratio per naming-convention family, across the whole
    original dataset (not deduped - a duplicate's brightness is a real data point for
    whichever family's filename it happened to carry)."""
    df["family"] = df["filename"].apply(naming_family)
    fam = df.groupby("family").agg(
        n=("cancer", "size"),
        cancer=("cancer", "sum"),
        mean_brightness=("brightness", "mean"),
    )
    fam["cancer_ratio"] = fam["cancer"] / fam["n"]
    fam = fam.sort_values("n", ascending=False)
    pd.set_option("display.width", 120)
    print(fam.to_string(float_format=lambda x: f"{x:.3f}"))

    skewed = fam[(fam["n"] >= 20) & ((fam["cancer_ratio"] >= 0.95) | (fam["cancer_ratio"] <= 0.05))]
    print(f"\nFamilies with n>=20 that are >=95% one class:")
    if len(skewed):
        print(skewed.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        print("  none")
    return fam


def main():
    df = load_all_rows()
    print(f"Loaded {len(df)} rows across train/valid/test.\n")

    print("Hashing + measuring brightness for every image (single pass)...")
    hashes, brightness, missing = hash_and_brightness(df)
    if missing:
        print(f"  WARNING: {len(missing)} files listed in a _classes.csv were not found on disk, dropping them")
        df = df.drop(df.index[missing]).reset_index(drop=True)
        hashes = np.delete(hashes, missing, axis=0)
        brightness = np.delete(brightness, missing, axis=0)
    df["brightness"] = brightness

    print("\n=== Naming-family / brightness / cancer-ratio breakdown (all original images) ===")
    report_family_brightness(df)

    print("\n=== Perceptual-hash near-duplicate grouping ===")
    group_ids = build_dup_groups(hashes)
    df["group_id"] = group_ids
    n_groups = df["group_id"].nunique()
    group_sizes = df.groupby("group_id").size()
    print(f"\n{len(df)} images -> {n_groups} unique phash groups "
          f"(largest group: {group_sizes.max()} images, "
          f"{(group_sizes > 1).sum()} groups have 2+ images)")

    print("\n=== Reassigning groups to splits ===")
    df, counts, cancer_counts, inconsistent, overall_rate = assign_groups(df)
    print(f"Overall cancer rate: {overall_rate:.4f}")
    if inconsistent:
        print(f"WARNING: {len(inconsistent)} groups have inconsistent cancer labels across near-duplicates "
              f"(majority used for balancing only).")
    else:
        print("No label inconsistencies found within duplicate groups.")

    print("\nNew split sizes:")
    for split in SPLITS:
        n = counts[split]
        c = cancer_counts[split]
        print(f"  {split}: {n} images ({n/len(df):.1%} of total) - {c} cancer / {n-c} normal "
              f"(cancer rate {c/n:.4f})")

    leak_check = df.groupby("group_id")["new_split"].nunique()
    assert (leak_check == 1).all(), "a duplicate group ended up split across multiple new splits"
    print("\nSanity check passed: every near-duplicate group is fully contained in one split.")

    os.makedirs(os.path.join(base, "splits_fixed"), exist_ok=True)
    for split in SPLITS:
        out = df[df["new_split"] == split][["filename", "source_split", "cancer", "normal"]]
        out.to_csv(os.path.join(base, "splits_fixed", f"{split}.csv"), index=False)
        print(f"Wrote splits_fixed/{split}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
