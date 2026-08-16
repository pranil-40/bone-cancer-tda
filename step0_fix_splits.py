# step0: fixes the train/valid/test leakage found during review - Roboflow's
# export put multiple copies of the same source X-ray (same base filename,
# different .rf.<hash> export id) into different splits. This groups every
# image by its base source name across the WHOLE dataset and reassigns each
# group to a single split, so no source image ever appears on both sides of
# a train/test boundary. Downstream scripts (step8b, step9c, step10) read
# the resulting splits_fixed/*.csv manifests instead of the original folder
# CSVs.

import os
import re
import pandas as pd

base = r"C:\projects\bone_cancer_tda"
SPLITS = ["train", "valid", "test"]

# original dataset proportions - what we're trying to preserve after reassignment
TARGET_PROPORTIONS = {"train": 7057 / 8811, "valid": 882 / 8811, "test": 872 / 8811}


def base_name(filename):
    """Strip Roboflow's '.rf.<hash>.<ext>' export suffix to recover the source image name."""
    return re.sub(r"\.rf\.[0-9a-f]{16,}\.(jpg|jpeg|png)$", "", filename.strip(), flags=re.I)


def load_all_rows():
    """Read all three original _classes.csv files into one frame, tagging each row with source_split."""
    frames = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
        df.columns = df.columns.str.strip()
        df["source_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_groups(df):
    """Group rows by base source-image name, then greedily assign whole groups to
    train/valid/test so no group crosses a split boundary, keeping each split's
    image share close to TARGET_PROPORTIONS and its cancer rate close to the
    dataset-wide cancer rate."""
    df["base"] = df["filename"].apply(base_name)

    total_images = len(df)
    total_cancer = df["cancer"].sum()
    overall_cancer_rate = total_cancer / total_images

    groups = df.groupby("base")
    # biggest groups first - placing them early keeps the greedy packing closer to target
    ordered = sorted(groups, key=lambda kv: (-len(kv[1]), kv[0]))

    target_count = {s: TARGET_PROPORTIONS[s] * total_images for s in SPLITS}
    target_cancer = {s: target_count[s] * overall_cancer_rate for s in SPLITS}

    counts = {s: 0 for s in SPLITS}
    cancer_counts = {s: 0 for s in SPLITS}
    inconsistent_groups = []

    assignment = {}  # base name -> new split
    for base_val, rows in ordered:
        labels = set(rows["cancer"])
        if len(labels) > 1:
            inconsistent_groups.append(base_val)
        gsize = len(rows)
        gcancer = rows["cancer"].sum()

        # give the group to whichever split is furthest below its own target,
        # in relative terms, so a small-target split (valid/test) isn't treated
        # as "full" just because its absolute deficit is small
        best_split, best_score = None, float("-inf")
        for split in SPLITS:
            size_deficit = (target_count[split] - counts[split]) / target_count[split]
            cancer_deficit = (target_cancer[split] - cancer_counts[split]) / max(target_cancer[split], 1)
            score = size_deficit + cancer_deficit
            if score > best_score:
                best_split, best_score = split, score

        counts[best_split] += gsize
        cancer_counts[best_split] += gcancer
        assignment[base_val] = best_split

    df["new_split"] = df["base"].map(assignment)
    return df, counts, cancer_counts, inconsistent_groups, overall_cancer_rate


def main():
    df = load_all_rows()
    df, counts, cancer_counts, inconsistent, overall_rate = assign_groups(df)

    n_groups = df["base"].nunique()
    print(f"Total rows: {len(df)}  |  unique source images (groups): {n_groups}")
    print(f"Overall cancer rate: {overall_rate:.4f}")
    if inconsistent:
        print(f"\nWARNING: {len(inconsistent)} groups have inconsistent cancer labels across duplicates "
              f"(same base image, different label) - not auto-resolved, using majority for balancing only.")
        for b in inconsistent[:10]:
            print(f"  {b}")
    else:
        print("\nNo label inconsistencies found within duplicate groups.")

    print("\nNew split sizes:")
    for split in SPLITS:
        n = counts[split]
        c = cancer_counts[split]
        print(f"  {split}: {n} images ({n/len(df):.1%} of total) - {c} cancer / {n-c} normal "
              f"(cancer rate {c/n:.4f})")

    # sanity check: confirm no group landed in more than one new split
    leak_check = df.groupby("base")["new_split"].nunique()
    assert (leak_check == 1).all(), "a base image group ended up split across multiple new splits"
    print("\nSanity check passed: every source-image group is fully contained in one split.")

    os.makedirs(os.path.join(base, "splits_fixed"), exist_ok=True)
    for split in SPLITS:
        out = df[df["new_split"] == split][["filename", "source_split", "cancer", "normal"]]
        out.to_csv(os.path.join(base, "splits_fixed", f"{split}.csv"), index=False)
        print(f"Wrote splits_fixed/{split}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
