# first try at fixing the leakage. Roboflow put copies of the same image (same
# base filename, different .rf.<hash>) in different splits, so this groups by
# base filename and gives each group to one split.
# replaced by split_phash.py, kept for the record

import os
import re
import pandas as pd

base = r"C:\projects\bone_cancer_tda"
SPLITS = ["train", "valid", "test"]

# original proportions
TARGET_PROPORTIONS = {"train": 7057 / 8811, "valid": 882 / 8811, "test": 872 / 8811}


def base_name(filename):
    # strip the .rf.<hash> suffix to get the source image name
    return re.sub(r"\.rf\.[0-9a-f]{16,}\.(jpg|jpeg|png)$", "", filename.strip(), flags=re.I)


def load_all_rows():
    # all three original _classes.csv files in one dataframe
    frames = []
    for split in SPLITS:
        df = pd.read_csv(os.path.join(base, split, "_classes.csv"))
        df.columns = df.columns.str.strip()
        df["source_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_groups(df):
    # greedy: biggest groups first, each one goes to whichever split is furthest below
    # its target size and cancer count
    df["base"] = df["filename"].apply(base_name)

    total_images = len(df)
    total_cancer = df["cancer"].sum()
    overall_cancer_rate = total_cancer / total_images

    groups = df.groupby("base")
    # biggest groups first
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

        # furthest below target wins (relative, so valid/test don't look full early)
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
        print(f"\n{len(inconsistent)} groups have mixed cancer labels across duplicates "
              f"(same base image, different label), using the majority for balancing")
        for b in inconsistent[:10]:
            print(f"  {b}")
    else:
        print("\nNo label inconsistencies found within duplicate groups.")

    print("\nNew split sizes:")
    for split in SPLITS:
        n = counts[split]
        c = cancer_counts[split]
        print(f"  {split}: {n} images ({n/len(df):.1%} of total), {c} cancer / {n-c} normal "
              f"(cancer rate {c/n:.4f})")

    # no group should end up in two splits
    leak_check = df.groupby("base")["new_split"].nunique()
    assert (leak_check == 1).all(), "a base image group ended up split across multiple new splits"
    print("\nCheck passed: every source-image group is in one split.")

    os.makedirs(os.path.join(base, "1-Data", "splits"), exist_ok=True)
    for split in SPLITS:
        out = df[df["new_split"] == split][["filename", "source_split", "cancer", "normal"]]
        out.to_csv(os.path.join(base, "1-Data", "splits", f"{split}.csv"), index=False)
        print(f"Wrote 1-Data/splits/{split}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
