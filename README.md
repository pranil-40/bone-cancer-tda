# Bone Cancer X-ray Classification with Persistent Homology

Comparing topological features (Betti curves from persistent homology, via GUDHI)
against a plain CNN transfer-learning baseline for classifying bone X-rays as
cancerous vs. normal. Dataset is the [Bone Cancer Detection](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru)
set from Roboflow (8,811 images, CC BY 4.0).

## Data leakage: found and fixed

The train/valid/test split that ships with this dataset has **duplicate source
images across splits** — stripping Roboflow's `.rf.<hash>` export suffix and
comparing base filenames, 8,811 rows collapsed to 5,303 unique source images
(some images appeared 10+ times in `train/` alone, under different export
hashes), with 367 of 781 unique `test/` images also present in `train/`, plus
402 shared between train/valid and 123 between valid/test.

`step0_fix_splits.py` fixes this: it groups every row by base source-image
name across the *whole* dataset, then reassigns each group as a unit to a
single split (train/valid/test), so no source image can ever appear on both
sides of a train/test boundary. Group assignment is greedy — largest groups
placed first — targeting the original ~80/10/10 split proportions and the
dataset-wide cancer rate. Output is `splits_fixed/{train,valid,test}.csv`,
each row keeping a `source_split` column pointing back to wherever the actual
image file lives on disk.

Regrouped split sizes (cancer rate in parentheses):

| Split | Images | % of total | Cancer / Normal |
|---|---|---|---|
| train | 7,066 | 80.2% | 3,090 / 3,976 (0.4373) |
| valid | 877 | 10.0% | 389 / 488 (0.4436) |
| test | 868 | 9.9% | 384 / 484 (0.4424) |

Matches the original ~80/10/10 proportions closely, and class balance held up
well across all three (overall dataset cancer rate: 0.4384). No duplicate
group had inconsistent cancer labels across its copies, and a post-hoc check
confirms every group is fully contained in one split.

**What the leakage was actually worth**, comparing the same tuned classifiers
on the old (leaky) vs. corrected splits:

| | AUC, old → corrected (128x128) | AUC, old → corrected (224x224) |
|---|---|---|
| Random Forest | 0.9824 → 0.9731 | 0.9824 → 0.9725 |
| XGBoost | 0.9822 → 0.9747 | 0.9813 → 0.9757 |
| MLP | 0.9706 → 0.9555 | 0.9781 → 0.9620 |

So the leakage was real and inflated every classifier by roughly 1–3 points
of accuracy/AUC (more on accuracy than AUC) — but that's not the whole story:
performance is still genuinely strong after the fix (AUC 0.95–0.98), so
there's real topological signal in the Betti features beyond the duplicate
artifact. The CNN baselines moved by a similar, smaller amount, in mixed
directions (see below) — some of that is ordinary run-to-run noise, since
`step10_cnn_models.py` doesn't fix a `torch` random seed.

`step2`/`step6`/`step8_betti_resolution.py` and the original `betti_vectors_bone*.npz`
files are left as-is, as the historical record of the leaky run. Everything
downstream of the fix (`step8b`, `step9c`, `step10`) reads from `splits_fixed/`
instead.

## Pipeline order

| Step | Script | Produces |
|---|---|---|
| 0 | `step0_fix_splits.py` | `splits_fixed/{train,valid,test}.csv` — deduplicated, leakage-free splits (see above) |
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
| 0→8b | `step8b_betti_resolution_fixed.py` | `betti_vectors_bone_{128,224}_fixed.npz` — same Betti-curve math as step8, reading from `splits_fixed/` |
| 0→9c | `step9c_final_eval_fixed.py` | Test-set metrics at 128x128/224x224 on the corrected splits — same tuned hyperparameters, not retuned |
| 10 | `step10_check_torch.py`, `step10_cnn_models.py` | Environment check, then CNN transfer-learning baseline (reads `splits_fixed/`) |

`.npz` feature files, `train/`, `valid/`, `test/`, `venv/`, and `archive.zip`
are all gitignored — regenerate the features by running step2 (and step6/step8
for the min-life/resolution variants, or step0 + step8b for the corrected
splits).

## Classifiers

Betti vectors are standardized (`StandardScaler`, fit on train only) before
going into any of these. Hyperparameters below were selected in `step4_tuning.py`
against the (leaky) validation split and are the ones used in every eval script,
including `step9c` on the corrected splits — none of this was retuned after
the leakage fix, so the comparison in the table above isolates the effect of
the split fix alone.

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, min_samples_split=2, max_features="log2", class_weight="balanced"` |
| XGBoost | `n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8` |
| MLP | `hidden_layer_sizes=(128, 64), solver="adam", max_iter=500` |

All three use `random_state=42`.

### Results on corrected splits (step9c)

| Resolution | Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 128x128 | Random Forest | 0.9032 | 0.9731 | 0.8945 | 0.8641 | 0.9271 | 0.9271 | 0.8843 |
| 128x128 | XGBoost | 0.8998 | 0.9747 | 0.8892 | 0.8703 | 0.9089 | 0.9089 | 0.8926 |
| 128x128 | MLP | 0.9078 | 0.9555 | 0.8972 | 0.8858 | 0.9089 | 0.9089 | 0.9070 |
| 224x224 | Random Forest | 0.9147 | 0.9725 | 0.9061 | 0.8837 | 0.9297 | 0.9297 | 0.9029 |
| 224x224 | XGBoost | 0.9159 | 0.9757 | 0.9063 | 0.8937 | 0.9193 | 0.9193 | 0.9132 |
| 224x224 | MLP | 0.9159 | 0.9620 | 0.9058 | 0.8977 | 0.9141 | 0.9141 | 0.9174 |

## CNN baseline (step10)

Four ImageNet-pretrained architectures — ResNet-18, ResNet-50, GoogLeNet, and
EfficientNet-B0 — used as fixed feature extractors: every parameter is frozen
except the final classifier layer (`fc` / `classifier`), which is replaced with
a single linear output and trained for 3 epochs with Adam (`lr=1e-3`) and
`BCEWithLogitsLoss`. Images are resized to 128x128 and normalized with standard
ImageNet mean/std. Runs CPU-only — no CUDA available in this environment (see
`step10_check_torch.py`). Reads from `splits_fixed/`, so train/test no longer
share duplicate source images.

### Results on corrected splits, 128x128

| Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.8606 | 0.9423 | 0.8326 | 0.8879 | 0.7839 | 0.7839 | 0.9215 |
| ResNet-50 | 0.8733 | 0.9502 | 0.8568 | 0.8568 | 0.8568 | 0.8568 | 0.8864 |
| GoogLeNet | 0.8756 | 0.9439 | 0.8430 | 0.9539 | 0.7552 | 0.7552 | 0.9711 |
| EfficientNet-B0 | 0.8675 | 0.9441 | 0.8568 | 0.8210 | 0.8958 | 0.8958 | 0.8450 |

## A couple of things worth knowing about the features

- The "4 channels" in the Betti vectors (grayscale mean, R, G, B) are mostly
  redundant: since the source X-rays are grayscale saved as RGB, R, G, and B
  are pixel-identical, so 600 of the 800 feature dimensions duplicate the
  grayscale channel's information.
- `step4_tuning.py` reflects only the second of two tuning rounds that were
  run — the first round's file was overwritten before it was ever committed,
  so its exact sweep values aren't recoverable.
