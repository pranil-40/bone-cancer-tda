# step11: radiomics feature extraction (pyradiomics) on the corrected,
# phash-deduped splits from step0c_phash_dedup.py - a second, independent
# feature family to compare against the Betti-curve pipeline. Extracts
# first-order, GLCM, GLRLM, GLSZM, and 2D shape features per Mayerhoefer et
# al.'s standard radiomics feature set.
#
# LIMITATION, stated explicitly per the request that produced this script:
# this dataset has no tumor segmentation mask, so the ROI used here is
# effectively the WHOLE IMAGE - pyradiomics itself refuses a mask that's
# 100% foreground (it requires at least one background-labeled pixel to
# recognize a mask as "segmented" at all, see radiomics.imageoperations.
# getMask), so the actual ROI is the whole frame minus a 2px border, which
# is a no-op on the science and purely there to satisfy that check. Real
# radiomics is built around a segmented lesion - using the full frame means
# first-order/texture stats are diluted by background and non-lesion
# anatomy, and the shape2D features are close to meaningless: since every
# image is resized to the same SIZE x SIZE canvas and the mask is
# "everything but a 2px border", shape2D descriptors (area, perimeter,
# elongation, etc.) come out nearly identical across every image - they
# describe the canvas, not the pathology. They're included anyway because
# they were asked for, and the near-zero variance IS the demonstration of
# the limitation.

import logging
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from radiomics import featureextractor
import radiomics

base = r"C:\projects\bone_cancer_tda"
SIZE = 224
FEATURE_CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "shape2D"]

radiomics.logger.setLevel(logging.ERROR)  # pyradiomics is very chatty per-image by default

_extractor = None


def _init_worker():
    """Build one extractor per worker process - pyradiomics extractors aren't picklable
    across the multiprocessing.Pool boundary, so each process makes its own."""
    global _extractor
    ex = featureextractor.RadiomicsFeatureExtractor(force2D=True, force2Ddimension=0)
    ex.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    _extractor = ex


def extract_one(args):
    """Extract radiomics features for one image, using the whole frame as the ROI
    (see module docstring). Returns (filename, cancer, feature_dict_or_None, error_or_None)."""
    filename, source_split, cancer = args
    path = os.path.join(base, source_split, filename)
    try:
        img = np.array(Image.open(path).convert("L").resize((SIZE, SIZE)), dtype=np.float64)
        img3d = img[np.newaxis, :, :]           # pyradiomics/SimpleITK expect a 3D (z,y,x) array
        mask3d = np.ones_like(img3d, dtype=np.uint8)
        mask3d[:, :2, :] = 0; mask3d[:, -2:, :] = 0    # pyradiomics requires >=1 background pixel
        mask3d[:, :, :2] = 0; mask3d[:, :, -2:] = 0    # to recognize the mask as segmented at all
        sitk_img = sitk.GetImageFromArray(img3d)
        sitk_mask = sitk.GetImageFromArray(mask3d)
        result = _extractor.execute(sitk_img, sitk_mask)
        feats = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}
        return filename, cancer, feats, None
    except Exception as e:
        return filename, cancer, None, f"{type(e).__name__}: {e}"


def process(split):
    """Extract features for every image in one corrected split, in parallel."""
    df = pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))
    args = list(zip(df["filename"], df["source_split"], df["cancer"]))
    print(f"{split}: extracting radiomics features for {len(args)} images at {SIZE}x{SIZE}...")

    start = time.time()
    with Pool(cpu_count(), initializer=_init_worker) as pool:
        results = pool.map(extract_one, args)
    print(f"{split}: done in {time.time()-start:.0f}s")

    failures = [(fn, err) for fn, _, feats, err in results if feats is None]
    ok = [(fn, cancer, feats) for fn, cancer, feats, err in results if feats is not None]
    if failures:
        print(f"  WARNING: {len(failures)}/{len(args)} images failed feature extraction:")
        for fn, err in failures[:20]:
            print(f"    {fn}: {err}")

    feature_names = sorted(ok[0][2].keys())
    rows = []
    for fn, cancer, feats in ok:
        missing = [k for k in feature_names if k not in feats]
        if missing:
            print(f"  WARNING: {fn} missing {len(missing)} features present elsewhere: {missing}")
        rows.append({"filename": fn, "cancer": cancer, **{k: feats.get(k, np.nan) for k in feature_names}})

    out_df = pd.DataFrame(rows)
    print(f"{split}: {out_df.shape[1]-2} features x {len(out_df)} images extracted "
          f"({len(failures)} images dropped for extraction failure)")
    return out_df


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        out_df = process(split)
        out_path = os.path.join(base, f"radiomics_features_{split}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} columns)\n")
