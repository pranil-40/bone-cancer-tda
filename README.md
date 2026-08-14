# Bone Cancer X-ray Classification with Persistent Homology

Comparing topological features (Betti curves from persistent homology, via GUDHI)
against a plain CNN transfer-learning baseline for classifying bone X-rays as
cancerous vs. normal. Dataset is the [Bone Cancer Detection](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru)
set from Roboflow (8,811 images, CC BY 4.0).

## ⚠️ Before trusting any AUC number here

The train/valid/test split that ships with this dataset has **duplicate source
images across splits**. Stripping Roboflow's `.rf.<hash>` suffix and comparing
base filenames: 367 of the 781 unique images in `test/` also appear in `train/`
(some images show up 10+ times in `train/` alone under different hashes), plus
402 shared between train/valid and 123 between valid/test. That's the kind of
thing that inflates AUC — and inflates it *more* at higher resolution, since
64x64 downsampling blurs out the pixel-level detail that lets a model recognize
a near-duplicate of a training image sitting in the test set. Any AUC jump
between the 64x64 and 128x128/224x224 runs should be read with this in mind —
it's not necessarily "topological features work better at higher res." Fixing
this means deduping by base filename and re-splitting at the source-image
level before the numbers mean much.

## Pipeline order

| Step | Script | Produces |
|---|---|---|
| 1 | `step1_explore.py` | Class counts per split, sanity-checks that images load |
| 2 | `step2_betti.py` / `step2_betti_fast.py` | `betti_vectors_bone.npz` — 800-dim Betti-curve features at 64x64 (fast version is a parallelized rewrite, same output) |
| 3 | `step3_models.py` | RF / XGBoost / MLP baseline metrics, default hyperparameters |
| 4 | `step4_tuning.py` | Manual hyperparameter sweeps against the validation split (second/final round — see note in the file) |
| 5 | `step5_best_models.py` | Test-set metrics using the tuned hyperparameters below |
| 6 | `step6_betti_minlife.py` | `betti_vectors_bone_minlife{0,5,10,20}.npz` — 64x64 features with short-lived persistence pairs filtered out |
| 7 | `step7_compare_minlife.py` | Validation AUC across the four `MIN_LIFE` settings |
| 8 | `step8_betti_resolution.py` | `betti_vectors_bone_{128,224}.npz` — same features at higher resolution |
| 9 | `step9_final_eval.py` | Test-set metrics at 128x128/224x224 |
| 9b | `step9b_final_eval_64.py` | Same eval as step9, run against the original 64x64 features, for a same-format comparison point |
| 10 | `step10_check_torch.py`, `step10_cnn_models.py` | Environment check, then CNN transfer-learning baseline (not a Betti-feature step) |

`.npz` feature files, `train/`, `valid/`, `test/`, `venv/`, and `archive.zip`
are all gitignored — regenerate the features by running step2 (and step6/step8
if you want the min-life or resolution variants).

## Classifiers

Betti vectors are standardized (`StandardScaler`, fit on train only) before
going into any of these. Hyperparameters below were selected in `step4_tuning.py`
against the validation split and are the ones used in `step5`/`step9`/`step9b`.

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, min_samples_split=2, max_features="log2", class_weight="balanced"` |
| XGBoost | `n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8` |
| MLP | `hidden_layer_sizes=(128, 64), solver="adam", max_iter=500` |

All three use `random_state=42`.

## CNN baseline (step10)

Four ImageNet-pretrained architectures — ResNet-18, ResNet-50, GoogLeNet, and
EfficientNet-B0 — used as fixed feature extractors: every parameter is frozen
except the final classifier layer (`fc` / `classifier`), which is replaced with
a single linear output and trained for 3 epochs with Adam (`lr=1e-3`) and
`BCEWithLogitsLoss`. Images are resized to 64x64 (matching the Betti-feature
resolution, not the 224x224 these architectures were originally trained on) and
normalized with standard ImageNet mean/std. Runs CPU-only — no CUDA available
in this environment (see `step10_check_torch.py`).

## A couple of things worth knowing about the features

- The "4 channels" in the Betti vectors (grayscale mean, R, G, B) are mostly
  redundant: since the source X-rays are grayscale saved as RGB, R, G, and B
  are pixel-identical, so 600 of the 800 feature dimensions duplicate the
  grayscale channel's information.
- `step4_tuning.py` reflects only the second of two tuning rounds that were
  run — the first round's file was overwritten before it was ever committed,
  so its exact sweep values aren't recoverable.
