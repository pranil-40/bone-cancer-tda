# same as extract.py with shape2D off (first-order plus glcm, glrlm, glszm, gldm,
# ngtdm). shape2D had no importance in the first run. same ROI problem as
# extract.py, so the texture stats are still diluted by background

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
FEATURE_CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]  # no shape2D

radiomics.logger.setLevel(logging.ERROR)

_extractor = None


def _init_worker():
    # every worker builds its own extractor, they can't be pickled
    global _extractor
    ex = featureextractor.RadiomicsFeatureExtractor(force2D=True, force2Ddimension=0)
    ex.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        ex.enableFeatureClassByName(fc)
    _extractor = ex


def extract_one(args):
    # returns (filename, cancer, features, error)
    filename, source_split, cancer = args
    path = os.path.join(base, source_split, filename)
    try:
        img = np.array(Image.open(path).convert("L").resize((SIZE, SIZE)), dtype=np.float64)
        img3d = img[np.newaxis, :, :]
        mask3d = np.ones_like(img3d, dtype=np.uint8)
        mask3d[:, :2, :] = 0; mask3d[:, -2:, :] = 0
        mask3d[:, :, :2] = 0; mask3d[:, :, -2:] = 0
        sitk_img = sitk.GetImageFromArray(img3d)
        sitk_mask = sitk.GetImageFromArray(mask3d)
        result = _extractor.execute(sitk_img, sitk_mask)
        feats = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}
        return filename, cancer, feats, None
    except Exception as e:
        return filename, cancer, None, f"{type(e).__name__}: {e}"


def process(split):
    df = pd.read_csv(os.path.join(base, "1-Data", "splits", f"{split}.csv"))
    args = list(zip(df["filename"], df["source_split"], df["cancer"]))
    print(f"{split}: extracting texture-only radiomics features for {len(args)} images at {SIZE}x{SIZE}...")

    start = time.time()
    with Pool(cpu_count(), initializer=_init_worker) as pool:
        results = pool.map(extract_one, args)
    print(f"{split}: done in {time.time()-start:.0f}s")

    failures = [(fn, err) for fn, _, feats, err in results if feats is None]
    ok = [(fn, cancer, feats) for fn, cancer, feats, err in results if feats is not None]
    if failures:
        print(f"  {len(failures)}/{len(args)} images failed:")
        for fn, err in failures[:20]:
            print(f"    {fn}: {err}")

    feature_names = sorted(ok[0][2].keys())
    rows = []
    for fn, cancer, feats in ok:
        missing = [k for k in feature_names if k not in feats]
        if missing:
            print(f"  {fn} is missing {len(missing)} features: {missing}")
        rows.append({"filename": fn, "cancer": cancer, **{k: feats.get(k, np.nan) for k in feature_names}})

    out_df = pd.DataFrame(rows)
    print(f"{split}: {out_df.shape[1]-2} features x {len(out_df)} images extracted "
          f"({len(failures)} images dropped for extraction failure)")
    return out_df


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        out_df = process(split)
        out_path = os.path.join(base, "2-Features", "radiomics", f"texture_{split}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({out_df.shape[0]} rows, {out_df.shape[1]} columns)\n")
