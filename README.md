# Bone Cancer X-ray Classification with Persistent Homology

This project compares topological features (Betti curves from persistent homology, computed with GUDHI) against radiomics features (pyradiomics) and a plain CNN transfer-learning baseline for classifying bone X-rays as cancerous or normal. The data is the [Bone Cancer Detection](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru) set from Roboflow (8,811 images, CC BY 4.0).

## Layout

```
1-Data/        splits, duplicate checks, exploration
  splits/        train/valid/test manifests (the deduplicated split everything uses)
2-Features/    feature extraction
  betti/         Betti curve vectors (64, 128, 224 px)
  radiomics/     pyradiomics features, plus the committed CSVs
  cnn/           ResNet-18 / ResNet-50 embeddings
3-Models/      classifiers and evaluation
  betti/         RF / XGBoost / MLP on Betti vectors
  cnn/           frozen-backbone transfer learning
  radiomics/     XGBoost on radiomics features
  fusion/        TDA + ResNet fusion models
```

Run every script from the repo root (for example `python 3-Models/betti/eval_size_dedup.py`). Each script has a hardcoded `base` path near the top that needs to point at your local copy of the repo. The image folders (`train/`, `valid/`, `test/`), `venv/`, `archive.zip` and all `.npz` files are gitignored. The radiomics CSVs are committed because they're small (1-10MB each) and slow to regenerate.

Intermediate `.npz` files are written to the repo root:

| File | Written by |
|---|---|
| `betti_64.npz` | `betti.py` / `betti_fast.py` |
| `betti_64_minlife{0,5,10,20}.npz` | `betti_minlife.py` |
| `betti_{128,224}_orig.npz` | `betti_size.py` (original leaky split) |
| `betti_{128,224}_dedup.npz` | `betti_size_dedup.py` (deduplicated split) |
| `resnet18_features_{train,valid,test}.npz` | `resnet18.py` |
| `resnet50_features_{train,valid,test}.npz` | `resnet50.py` |

## Data leakage: found and fixed in two rounds

**Round 1: filename dedup (`1-Data/split_filename.py`, now superseded).** The train/valid/test split that ships with the dataset has duplicate source images across splits. After stripping Roboflow's `.rf.<hash>` export suffix and comparing base filenames, 8,811 rows collapsed to 5,303 unique source images, and 367 of the 781 unique `test/` images were also in `train/`. The script grouped rows by base filename and reassigned whole groups to one split.

**Round 2: perceptual-hash dedup (`1-Data/split_phash.py`, current).** A follow-up check (`1-Data/dupes.py`) showed that filename dedup missed real duplicates. The same source X-ray sometimes shows up under a completely unrelated filename (I confirmed this by eye, for example `foot_8_2_png...` and `image-no47-normal-_png...` are the same photo). `split_phash.py` replaces the filename check with perceptual hashing (phash, hash_size=16, Hamming distance <= 10) across all three original splits at once, groups connected near-duplicates with union-find, and reassigns whole groups the same way. This is now the canonical split and it overwrites `1-Data/splits/`.

Known limitation of round 2: standard phash is not flip or rotation invariant, so a flipped or 90-degree-rotated copy of a training image can still slip through. The 5,990 unique groups found (out of 8,811 images) should be read as "at least this many duplicates."

Split sizes after phash dedup:

| Split | Images | % of total | Cancer / Normal | Cancer rate |
|---|---|---|---|---|
| train | 7,041 | 79.9% | 3,100 / 3,941 | 0.4403 |
| valid | 879 | 10.0% | 388 / 491 | 0.4414 |
| test | 891 | 10.1% | 375 / 516 | 0.4209 |

These are close to the original ~80/10/10 proportions and the dataset-wide cancer rate (0.4384). No duplicate group had inconsistent cancer labels, and every group sits entirely in one split.

Here is how much each round of fixing changed the results (same tuned classifiers, no retuning at any stage):

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

Every classifier dropped further after the phash dedup than after the filename-only fix, which is what you'd expect if round 1 left real leakage behind. AUC is still 0.92 to 0.97 after both rounds, which is strong, but read the next section before taking that as "the model learned bone cancer."

## Open issue: naming-family / source confound

I grouped images by filename "naming family" (`IMG0000xxx`, `bone-cancer_train_x`, `image-no###`, `tibia_osteosarcoma`, and so on) and found that nearly every family is 100% one class or 0% the other. Roughly two thirds of all cancer-labeled images come from a single `bone-cancer_train/valid/test` family. The rest come from families named after their diagnosis (`osteosarcoma`, `ewing`, `chondrosarcoma`...). The normal class is dominated by separately branded collections, and some of them (`istockphoto`, `chest`) aren't even the same imaging domain as the cancer examples. Mean brightness also differs sharply by class (cancer 81.0 vs. normal 60.9, Cohen's d around 0.63 in the train set), which looks like a downstream symptom of the same thing.

So the dataset is a merge of a cancer-specific source and a separately curated "normal" source, each with its own visual fingerprint (compression, framing, contrast) that has nothing to do with pathology. Re-splitting cleanly does not fix this. It only stops the model from memorizing individual images. A model can still learn "this contrast and framing means cancer," and that shortcut is present in train and test alike, so it generalizes across a clean split just as well as a leaky one.

**Status: flagged, not resolved.** I haven't decided yet whether to normalize per-source characteristics, drop the most extreme sub-sources, or just document this as a limitation of the dataset. Every AUC in this README above 0.92 should be read with this in mind, since it may partly reflect source detection instead of pathology detection.

## Pipeline order

| Folder | Script | What it does |
|---|---|---|
| 1-Data | `split_filename.py` | Filename-based split fix. Superseded by `split_phash.py`, kept as a record |
| 1-Data | `dupes.py` | Class brightness/size stats plus a phash near-duplicate spot check that led to the phash split |
| 1-Data | `split_phash.py` | Writes `splits/{train,valid,test}.csv`, the current leakage-free splits. Also prints the naming-family and brightness breakdown |
| 1-Data | `explore.py` | Class counts per split, and a check that images load |
| 2-Features/betti | `betti.py`, `betti_fast.py` | 800-dim Betti-curve features at 64x64 on the original (leaky) splits. The parallel version is a faster rewrite with the same output |
| 3-Models/betti | `baseline.py` | RF / XGBoost / MLP baseline metrics, default hyperparameters |
| 3-Models/betti | `tune.py` | Manual hyperparameter sweeps on the validation split (second and final round, see the note in the file) |
| 3-Models/betti | `tuned.py` | Test-set metrics with the tuned hyperparameters listed below |
| 2-Features/betti | `betti_minlife.py` | 64x64 features with short-lived persistence pairs filtered out (`MIN_LIFE` of 0, 5, 10, 20) |
| 3-Models/betti | `compare_minlife.py` | Validation AUC across the four `MIN_LIFE` settings |
| 2-Features/betti | `betti_size.py` | Same features at 128x128 and 224x224, on the original (leaky) splits |
| 3-Models/betti | `eval_size.py` | Test-set metrics at 128x128 / 224x224 on the leaky splits |
| 3-Models/betti | `eval_64.py` | Same eval on the 64x64 features, as a comparison point |
| 2-Features/betti | `betti_size_dedup.py` | Same Betti-curve math as `betti_size.py`, reading from the current `1-Data/splits/` |
| 3-Models/betti | `eval_size_dedup.py` | Test-set metrics at 128x128 / 224x224 on the current splits, same tuned hyperparameters, not retuned |
| 3-Models/cnn | `check_torch.py`, `cnn.py` | Environment check, then the CNN transfer-learning baseline (reads `1-Data/splits/`) |
| 2-Features/radiomics | `extract.py` | pyradiomics features (first-order, GLCM, GLRLM, GLSZM, shape2D) on the current splits |
| 3-Models/radiomics | `xgb.py` | XGBoost train, tune and eval on those features |
| 2-Features/radiomics | `extract_texture.py` | First-order plus texture only (adds GLDM and NGTDM, drops shape2D), 93 features |
| 3-Models/radiomics | `xgb_texture.py` | XGBoost on the texture-only features |
| 2-Features/cnn | `resnet18.py`, `resnet50.py` | Frozen ResNet embeddings (512-dim and 2048-dim) for the fusion models |
| 3-Models/fusion | `fusion18.py`, `fusion18_tuned.py`, `fusion50.py` | Feature-level fusion of Betti vectors and ResNet embeddings |
| 3-Models/fusion | `blend.py` | Prediction-level fusion (weighted blend and stacked logistic regression) |

To regenerate the gitignored files, run `betti_fast.py` (and `betti_minlife.py` / `betti_size.py` for the variants), or `split_phash.py` then `betti_size_dedup.py` for the corrected splits.

## Classifiers (Betti features)

Betti vectors are standardized (`StandardScaler`, fit on train only) before going into any of these. The hyperparameters below were picked in `tune.py` against the (leaky) validation split and are used in every eval script, including `eval_size_dedup.py` on the corrected splits. Nothing was retuned after either leakage fix, so the before/after comparisons above isolate the effect of the split fixes.

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, min_samples_split=2, max_features="log2", class_weight="balanced"` |
| XGBoost | `n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8` |
| MLP | `hidden_layer_sizes=(128, 64), solver="adam", max_iter=500` |

All three use `random_state=42`.

### Results on current (phash-deduped) splits (`eval_size_dedup.py`)

| Resolution | Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 128x128 | Random Forest | 0.8541 | 0.9444 | 0.8375 | 0.7882 | 0.8933 | 0.8933 | 0.8256 |
| 128x128 | XGBoost | 0.8754 | 0.9509 | 0.8590 | 0.8204 | 0.9013 | 0.9013 | 0.8566 |
| 128x128 | MLP | 0.8765 | 0.9438 | 0.8571 | 0.8354 | 0.8800 | 0.8800 | 0.8740 |
| 224x224 | Random Forest | 0.8653 | 0.9496 | 0.8504 | 0.7986 | 0.9093 | 0.9093 | 0.8333 |
| 224x224 | XGBoost | 0.8732 | 0.9515 | 0.8571 | 0.8149 | 0.9040 | 0.9040 | 0.8508 |
| 224x224 | MLP | 0.8822 | 0.9498 | 0.8613 | 0.8534 | 0.8693 | 0.8693 | 0.8915 |

## CNN baseline (`3-Models/cnn/cnn.py`)

Four ImageNet-pretrained architectures (ResNet-18, ResNet-50, GoogLeNet and EfficientNet-B0) are used as fixed feature extractors. Every parameter is frozen except the final classifier layer (`fc` / `classifier`), which is replaced with a single linear output and trained for 3 epochs with Adam (`lr=1e-3`) and `BCEWithLogitsLoss`. Images are resized to 128x128 and normalized with the standard ImageNet mean and std. It runs on CPU only, since there's no CUDA on my machine (see `check_torch.py`). The script doesn't fix a `torch` random seed, so rerunning it will shift these numbers slightly.

### Results on current (phash-deduped) splits, 128x128

| Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.8552 | 0.9390 | 0.8322 | 0.8122 | 0.8533 | 0.8533 | 0.8566 |
| ResNet-50 | 0.8810 | 0.9486 | 0.8490 | 0.9113 | 0.7947 | 0.7947 | 0.9438 |
| GoogLeNet | 0.8631 | 0.9345 | 0.8277 | 0.8799 | 0.7813 | 0.7813 | 0.9225 |
| EfficientNet-B0 | 0.8373 | 0.9182 | 0.8129 | 0.7875 | 0.8400 | 0.8400 | 0.8353 |

## Radiomics baseline

`extract.py` runs pyradiomics on the same `1-Data/splits/` data. It extracts first-order, GLCM, GLRLM, GLSZM and 2D shape features (Mayerhoefer et al. convention) at 224x224, giving 83 features in total with 0 extraction failures across all 8,811 images.

**Limitation:** the dataset has no tumor segmentation mask, so the ROI is effectively the whole image. pyradiomics refuses a mask that is 100% foreground (it needs at least one background pixel to treat the mask as segmented), so the actual ROI is the whole frame minus a 2px border. That's a workaround to get past the check, not a real region of interest. Proper radiomics is built around a segmented lesion. Using the full frame dilutes first-order and texture stats with background and normal anatomy, and it makes the shape2D features close to meaningless, since every image shares the same canvas size and an "everything" mask. Consistent with that, no shape2D feature makes the top-15 importances.

pyradiomics has no prebuilt Windows wheel and needs a C++ compiler to install (Microsoft C++ Build Tools). That's a one-time setup issue and has nothing to do with the code.

For XGBoost I started from the Betti-vector hyperparameters (`n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8`) as a reference point, which gave a validation AUC of 0.9628 right away. A one-at-a-time sweep around that point found a marginally better combination (`max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0`), but the gain was 0.9628 to 0.9630, which is noise. The Betti-tuned hyperparameters carried over almost unchanged to a feature space about 10x smaller than the Betti vectors.

### Test results (`max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0, n_estimators=200`)

| Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|
| 0.9080 | 0.9686 | 0.8932 | 0.8728 | 0.9147 | 0.9147 | 0.9031 |

Feature importance is dominated by GLCM texture (`DifferenceAverage` 0.346 and `Idm` 0.210, together over half of the total), followed by more GLCM, GLRLM and first-order features.

This AUC (0.9686) is in the same range as Betti (0.94 to 0.95) and the CNNs (0.92 to 0.95) on the same splits. I wouldn't read it as radiomics working better, since all three are equally exposed to the source confound described above. GLCM texture features are exactly the kind of per-source fingerprint (compression, contrast profile) that the confound could be hiding in.

## Fusion models

`3-Models/fusion/` has two kinds of fusion between the Betti vectors (128x128) and frozen ResNet embeddings. `fusion18.py` and `fusion50.py` project each branch to 32 dims, concatenate, and train a linear head (the ResNet-18 version reached AUC 0.9751 on test). `fusion18_tuned.py` adds hidden layers, BatchNorm, dropout and weight decay on top of that. `blend.py` instead combines the output probabilities of an XGBoost-on-Betti model and a ResNet-18 head, either with a validation-tuned blend weight or a stacked logistic regression. All of these use the same splits and the same source confound applies.

## Things worth knowing about the features

- The 4 "channels" in the Betti vectors (grayscale mean, R, G, B) are mostly redundant. The source X-rays are grayscale saved as RGB, so R, G and B are pixel-identical, and 600 of the 800 feature dimensions duplicate the grayscale channel.
- `tune.py` only reflects the second of two tuning rounds. The first round's file was overwritten before I committed it, so its exact sweep values can't be recovered.
