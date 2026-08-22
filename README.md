# Bone Cancer X-ray Classification with Persistent Homology

Comparing topological features (Betti curves from persistent homology, via GUDHI)
and radiomics features (pyradiomics) against a plain CNN transfer-learning
baseline, for classifying bone X-rays as cancerous vs. normal. Dataset is the
[Bone Cancer Detection](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru)
set from Roboflow (8,811 images, CC BY 4.0).

## Data leakage: found, and fixed in two rounds

**Round 1 — filename dedup (`step0_fix_splits.py`, superseded).** The
train/valid/test split that ships with this dataset has duplicate source
images across splits — stripping Roboflow's `.rf.<hash>` export suffix and
comparing base filenames, 8,811 rows collapsed to 5,303 unique source images,
with 367 of 781 unique `test/` images also present in `train/`. `step0_fix_splits.py`
grouped rows by base filename and reassigned whole groups to one split.

**Round 2 — perceptual-hash dedup (`step0c_phash_dedup.py`, current).** A
follow-up sanity check (`step0b_sanity_checks.py`) found that filename-based
dedup missed real duplicates: the same source X-ray sometimes reappears under
a completely unrelated filename (visually confirmed — e.g. `foot_8_2_png...`
and `image-no47-normal-_png...` are the same photo). `step0c_phash_dedup.py`
replaces the filename check with perceptual hashing (phash, hash_size=16,
Hamming distance ≤ 10) across *all three* original splits at once, groups
connected near-duplicates (via union-find), and reassigns whole groups the
same way. This is now the canonical split — it overwrites `splits_fixed/`.

Known limitation of round 2: standard phash is **not flip/rotation-invariant**,
so a flipped or 90°-rotated copy of a training image would still slip through
undetected. 5,990 unique groups found (of 8,811 images) should be read as "at
least this many duplicates," not a hard floor.

Current (phash-deduped) split sizes:

| Split | Images | % of total | Cancer / Normal | Cancer rate |
|---|---|---|---|---|
| train | 7,041 | 79.9% | 3,100 / 3,941 | 0.4403 |
| valid | 879 | 10.0% | 388 / 491 | 0.4414 |
| test | 891 | 10.1% | 375 / 516 | 0.4209 |

Close to the original ~80/10/10 proportions and the dataset-wide cancer rate
(0.4384). No duplicate group had inconsistent cancer labels, and every group
is fully contained in one split.

**What each round of fixing was actually worth** (same tuned classifiers,
no retuning, at each stage):

| | Original (leaky) | + filename dedup | + phash dedup (current) |
|---|---|---|---|
| RF AUC @ 128x128 | 0.9824 | 0.9731 | 0.9444 |
| XGB AUC @ 128x128 | 0.9822 | 0.9747 | 0.9509 |
| MLP AUC @ 128x128 | 0.9706 | 0.9555 | 0.9438 |
| RF AUC @ 224x224 | 0.9824 | 0.9725 | 0.9496 |
| XGB AUC @ 224x224 | 0.9813 | 0.9757 | 0.9515 |
| MLP AUC @ 224x224 | 0.9781 | 0.9620 | 0.9498 |
| ResNet-18 AUC (CNN, 128x128) | 0.9505 | 0.9423 | 0.9390 |
| ResNet-50 AUC (CNN, 128x128) | 0.9561 | 0.9502 | 0.9486 |
| GoogLeNet AUC (CNN, 128x128) | 0.9314 | 0.9439 | 0.9345 |
| EfficientNet-B0 AUC (CNN, 128x128) | 0.9273 | 0.9441 | 0.9182 |

Every classifier dropped further after the more aggressive phash dedup than
after the filename-only fix — exactly what you'd expect if round 1 had left
real leakage on the table. AUC is still 0.92–0.97 after both rounds, which is
strong, but see the next section before reading that as "the model learned
bone cancer."

## Open issue, not yet fixed: naming-family / source confound

A follow-up check (grouping images by filename "naming family" — `IMG0000xxx`,
`bone-cancer_train_x`, `image-no###`, `tibia_osteosarcoma`, etc.) found that
**nearly every naming family is 100% one class or 0% the other.** Roughly
two-thirds of all cancer-labeled images come from a single `bone-cancer_train/
valid/test` family; the rest come from families named directly after their
diagnosis (`osteosarcoma`, `ewing`, `chondrosarcoma`...). The normal class is
dominated by separately-branded collections, some of which (`istockphoto`,
`chest`) aren't even the same imaging domain as the cancer examples. Mean
brightness also differs sharply by class (cancer 81.0 vs. normal 60.9,
Cohen's d ≈ 0.63 in the train set) — a downstream symptom of the same issue,
not a separate one.

This means the dataset is a merge of a cancer-specific source and a
separately-curated "normal" source, each with its own visual fingerprint
(compression, framing, contrast) independent of pathology. **Re-splitting
cleanly (both rounds above) does not fix this** — it only stops the model
from memorizing individual images; it does nothing to stop it from learning
"this contrast/framing profile means cancer," because that shortcut is
present throughout train *and* test identically, and generalizes across a
clean split just as well as a leaky one.

**Status: flagged, not resolved.** Whether to normalize per-source
characteristics, exclude the purest problem sub-sources, or simply document
this as a hard dataset limitation is still an open decision. Every AUC number
in this README from 0.92 upward should be read with this in mind — it may
substantially reflect source detection rather than pathology detection.

## Pipeline order

| Step | Script | Produces |
|---|---|---|
| 0 | `step0_fix_splits.py` | Filename-based split fix — **superseded by step0c**, kept as historical record |
| 0b | `step0b_sanity_checks.py` | Diagnostic: class brightness/size stats + phash near-duplicate spot-check that motivated step0c |
| 0c | `step0c_phash_dedup.py` | `splits_fixed/{train,valid,test}.csv` — **current** canonical leakage-free splits (phash-based, see above); also reports the naming-family/brightness breakdown |
| 1 | `step1_explore.py` | Class counts per split, sanity-checks that images load |
| 2 | `step2_betti.py` / `step2_betti_fast.py` | `betti_vectors_bone.npz` — 800-dim Betti-curve features at 64x64, on the **original (leaky)** splits (fast version is a parallelized rewrite, same output) |
| 3 | `step3_models.py` | RF / XGBoost / MLP baseline metrics, default hyperparameters |
| 4 | `step4_tuning.py` | Manual hyperparameter sweeps against the validation split (second/final round — see note in the file) |
| 5 | `step5_best_models.py` | Test-set metrics using the tuned hyperparameters below |
| 6 | `step6_betti_minlife.py` | `betti_vectors_bone_minlife{0,5,10,20}.npz` — 64x64 features with short-lived persistence pairs filtered out |
| 7 | `step7_compare_minlife.py` | Validation AUC across the four `MIN_LIFE` settings |
| 8 | `step8_betti_resolution.py` | `betti_vectors_bone_{128,224}.npz` — same features at higher resolution, on the **original (leaky)** splits |
| 9 | `step9_final_eval.py` | Test-set metrics at 128x128/224x224, on the leaky splits |
| 9b | `step9b_final_eval_64.py` | Same eval as step9, run against the original 64x64 features, for a same-format comparison point |
| 8b | `step8b_betti_resolution_fixed.py` | `betti_vectors_bone_{128,224}_fixed.npz` — same Betti-curve math as step8, reading from the current `splits_fixed/` |
| 9c | `step9c_final_eval_fixed.py` | Test-set metrics at 128x128/224x224 on the current corrected splits — same tuned hyperparameters, not retuned |
| 10 | `step10_check_torch.py`, `step10_cnn_models.py` | Environment check, then CNN transfer-learning baseline (reads `splits_fixed/`) |
| 11 | `step11_radiomics_extract.py` | `radiomics_features_{train,valid,test}.csv` — pyradiomics features on `splits_fixed/` |
| 12 | `step12_radiomics_eval.py` | XGBoost train/tune/eval on the radiomics features |

`.npz` feature files, `train/`, `valid/`, `test/`, `venv/`, and `archive.zip`
are all gitignored — regenerate them by running step2 (and step6/step8 for the
min-life/resolution variants, or step0c + step8b for the corrected splits).
`radiomics_features_*.csv` are **not** gitignored and are committed directly —
they're small (~1-10MB each) and worth keeping for reproducibility rather than
re-running the extraction.

## Classifiers (Betti features)

Betti vectors are standardized (`StandardScaler`, fit on train only) before
going into any of these. Hyperparameters below were selected in `step4_tuning.py`
against the (leaky) validation split and are the ones used in every eval script,
including `step9c` on the corrected splits — none of this was retuned after
either leakage fix, so the before/after comparisons above isolate the effect
of the split fixes alone.

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, min_samples_split=2, max_features="log2", class_weight="balanced"` |
| XGBoost | `n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8` |
| MLP | `hidden_layer_sizes=(128, 64), solver="adam", max_iter=500` |

All three use `random_state=42`.

### Results on current (phash-deduped) splits (step9c)

| Resolution | Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 128x128 | Random Forest | 0.8541 | 0.9444 | 0.8375 | 0.7882 | 0.8933 | 0.8933 | 0.8256 |
| 128x128 | XGBoost | 0.8754 | 0.9509 | 0.8590 | 0.8204 | 0.9013 | 0.9013 | 0.8566 |
| 128x128 | MLP | 0.8765 | 0.9438 | 0.8571 | 0.8354 | 0.8800 | 0.8800 | 0.8740 |
| 224x224 | Random Forest | 0.8653 | 0.9496 | 0.8504 | 0.7986 | 0.9093 | 0.9093 | 0.8333 |
| 224x224 | XGBoost | 0.8732 | 0.9515 | 0.8571 | 0.8149 | 0.9040 | 0.9040 | 0.8508 |
| 224x224 | MLP | 0.8822 | 0.9498 | 0.8613 | 0.8534 | 0.8693 | 0.8693 | 0.8915 |

## CNN baseline (step10)

Four ImageNet-pretrained architectures — ResNet-18, ResNet-50, GoogLeNet, and
EfficientNet-B0 — used as fixed feature extractors: every parameter is frozen
except the final classifier layer (`fc` / `classifier`), which is replaced with
a single linear output and trained for 3 epochs with Adam (`lr=1e-3`) and
`BCEWithLogitsLoss`. Images are resized to 128x128 and normalized with standard
ImageNet mean/std. Runs CPU-only — no CUDA available in this environment (see
`step10_check_torch.py`). Reads from `splits_fixed/`. Note: the script doesn't
fix a `torch` random seed, so re-running it will shift these numbers slightly
run to run.

### Results on current (phash-deduped) splits, 128x128

| Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.8552 | 0.9390 | 0.8322 | 0.8122 | 0.8533 | 0.8533 | 0.8566 |
| ResNet-50 | 0.8810 | 0.9486 | 0.8490 | 0.9113 | 0.7947 | 0.7947 | 0.9438 |
| GoogLeNet | 0.8631 | 0.9345 | 0.8277 | 0.8799 | 0.7813 | 0.7813 | 0.9225 |
| EfficientNet-B0 | 0.8373 | 0.9182 | 0.8129 | 0.7875 | 0.8400 | 0.8400 | 0.8353 |

## Radiomics baseline (step11/step12)

`step11_radiomics_extract.py` runs pyradiomics on the same `splits_fixed/`
splits, extracting first-order, GLCM, GLRLM, GLSZM, and 2D shape features
(Mayerhoefer et al. convention) at 224x224 — **83 features total**, 0
extraction failures across all 8,811 images.

**Limitation, stated explicitly:** this dataset has no tumor segmentation
mask, so the ROI is effectively the whole image. pyradiomics actually refuses
a mask that's 100% foreground outright (it requires at least one
background-labeled pixel to recognize a mask as "segmented"), so the real ROI
is the whole frame minus a 2px border — a no-op workaround, not a real
region of interest. Proper radiomics is built around a segmented lesion;
using the full frame dilutes first-order/texture stats with background and
non-lesion anatomy, and makes the shape2D features close to meaningless,
since every image shares the same canvas size and an "everything" mask —
confirmed below, where no shape2D feature makes the top-15 importances.

pyradiomics has no prebuilt Windows wheel and needs a C++ compiler to install
(Microsoft C++ Build Tools) — a one-time environment setup, not a code issue.

XGBoost, starting from the Betti-vector's tuned hyperparameters
(`n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6,
colsample_bytree=0.8`) as a reference point: validation AUC 0.9628 straight
off. A one-at-a-time sweep around that point found a marginally better
combination (`max_depth=9, learning_rate=0.1, subsample=0.8,
colsample_bytree=1.0`), but the gain was 0.9628 → 0.9630 — noise-level, not
a meaningful improvement. The Betti-tuned hyperparameters transferred
essentially unchanged to an 83-feature space ~10x smaller than the Betti
vectors.

### Test results (final config: `max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0, n_estimators=200`)

| Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|
| 0.9080 | 0.9686 | 0.8932 | 0.8728 | 0.9147 | 0.9147 | 0.9031 |

Top feature importances are dominated by GLCM texture (`DifferenceAverage`
0.346, `Idm` 0.210 — together over half of total importance), then more
GLCM/GLRLM/first-order features. As predicted by the whole-image-ROI
limitation above, **no `shape2D` feature appears in the top 15.**

This AUC (0.9686) is in the same range as Betti (0.94–0.95) and CNN
(0.92–0.95) on the same splits — not evidence that radiomics "works better,"
since all three are equally exposed to the naming-family/source confound
described above. GLCM texture features in particular are exactly the kind of
per-source fingerprint (compression, contrast profile) that confound could be
hiding in.

## A couple of things worth knowing about the features

- The "4 channels" in the Betti vectors (grayscale mean, R, G, B) are mostly
  redundant: since the source X-rays are grayscale saved as RGB, R, G, and B
  are pixel-identical, so 600 of the 800 feature dimensions duplicate the
  grayscale channel's information.
- `step4_tuning.py` reflects only the second of two tuning rounds that were
  run — the first round's file was overwritten before it was ever committed,
  so its exact sweep values aren't recoverable.
