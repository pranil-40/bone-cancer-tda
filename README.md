# Bone Cancer X-ray Classification with Persistent Homology

Comparing topological features (Betti curves from persistent homology, using GUDHI) with radiomics features (pyradiomics) and a plain CNN transfer-learning baseline for telling cancerous bone X-rays from normal ones. The data is the [Bone Cancer Detection](https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru) set from Roboflow (8,811 images, CC BY 4.0).

## Folders

```
1-Data/        splits, duplicate checks, exploration
  splits/        train/valid/test manifests used by everything else
2-Features/    feature extraction
  betti/         Betti curve vectors (64, 128, 224 px)
  radiomics/     pyradiomics features and the committed CSVs
  cnn/           ResNet-18 / ResNet-50 embeddings
3-Models/      classifiers
  betti/         RF / XGBoost / MLP on the Betti vectors
  cnn/           frozen-backbone transfer learning
  radiomics/     XGBoost on the radiomics features
  fusion/        Betti + ResNet fusion
```

Run scripts from the repo root, like `python 3-Models/betti/eval_size_dedup.py`. Each script has a `base` path at the top that needs to point at your copy of the repo. The image folders (`train/`, `valid/`, `test/`), `venv/`, `archive.zip` and the `.npz` files are gitignored. The radiomics CSVs are committed since they're small and slow to regenerate.

The `.npz` files get written to the repo root:

| File | Written by |
|---|---|
| `betti_64.npz` | `betti.py` / `betti_fast.py` |
| `betti_64_minlife{0,5,10,20}.npz` | `betti_minlife.py` |
| `betti_{128,224}_orig.npz` | `betti_size.py` (original leaky split) |
| `betti_{128,224}_dedup.npz` | `betti_size_dedup.py` (deduped split) |
| `resnet18_features_{train,valid,test}.npz` | `resnet18.py` |
| `resnet50_features_{train,valid,test}.npz` | `resnet50.py` |

## Data leakage

I fixed this in two rounds.

**Round 1, filename dedup (`1-Data/split_filename.py`, no longer used).** The train/valid/test split that comes with the dataset has the same source images in different splits. After stripping Roboflow's `.rf.<hash>` suffix and comparing base filenames, the 8,811 rows turn into 5,303 unique source images, and 367 of the 781 unique `test/` images were also in `train/`. The script grouped rows by base filename and gave each whole group to one split.

**Round 2, perceptual hash dedup (`1-Data/split_phash.py`, current).** `1-Data/dupes.py` showed that filename dedup missed duplicates. The same X-ray sometimes shows up under a completely unrelated filename (I checked by eye, `foot_8_2_png...` and `image-no47-normal-_png...` are the same photo). `split_phash.py` uses perceptual hashing instead (phash, hash_size=16, Hamming distance <= 10) across all three original splits at once, links near-duplicates with union-find, and gives each whole group to one split. This overwrites `1-Data/splits/` and is the split used everywhere now.

Phash isn't flip or rotation invariant, so a flipped or rotated copy of a training image can still get through. The 5,990 unique groups (out of 8,811 images) are a lower bound on how many duplicates there are.

Split sizes after phash dedup:

| Split | Images | % of total | Cancer / Normal | Cancer rate |
|---|---|---|---|---|
| train | 7,041 | 79.9% | 3,100 / 3,941 | 0.4403 |
| valid | 879 | 10.0% | 388 / 491 | 0.4414 |
| test | 891 | 10.1% | 375 / 516 | 0.4209 |

That's close to the original ~80/10/10 and to the overall cancer rate (0.4384). No duplicate group had mixed labels, and every group is in a single split.

AUC at each stage, same tuned classifiers and no retuning:

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

Everything dropped more after the phash dedup than after the filename one, so the first fix had clearly left leakage behind. AUC is still 0.92 to 0.97, which is high, but see the next section before reading much into it.

## Source confound (not fixed)

I grouped the images by filename "naming family" (`IMG0000xxx`, `bone-cancer_train_x`, `image-no###`, `tibia_osteosarcoma` and so on). Almost every family is 100% one class or 0% the other. About two thirds of the cancer images come from one `bone-cancer_train/valid/test` family, and the rest come from families named after a diagnosis (`osteosarcoma`, `ewing`, `chondrosarcoma`...). The normal class is mostly separate branded collections, and a few of them (`istockphoto`, `chest`) aren't even the same kind of image as the cancer ones. Mean brightness also differs a lot by class (cancer 81.0, normal 60.9, Cohen's d around 0.63 in train), probably for the same reason.

So the dataset looks like a cancer-specific source merged with a separate "normal" source, and each has its own look (compression, framing, contrast) that has nothing to do with pathology. A clean re-split doesn't help with this. It only stops the model from memorizing single images. The model can still learn "this contrast and framing means cancer", and that works just as well on a clean split because the pattern is in train and test alike.

I haven't decided what to do about it yet. Options are normalizing per source, dropping the worst sub-sources, or just documenting it as a limit of the dataset. Until then, any AUC above 0.92 here might partly be source detection instead of cancer detection.

## Scripts

| Folder | Script | What it does |
|---|---|---|
| 1-Data | `split_filename.py` | Filename-based split fix. Replaced by `split_phash.py`, kept for the record |
| 1-Data | `dupes.py` | Brightness/size stats by class, plus a phash near-duplicate check that led to the phash split |
| 1-Data | `split_phash.py` | Writes `splits/{train,valid,test}.csv`. Also prints the naming-family and brightness breakdown |
| 1-Data | `explore.py` | Class counts per split, checks that images load |
| 2-Features/betti | `betti.py`, `betti_fast.py` | 800-dim Betti-curve features at 64x64 on the original (leaky) splits. The fast one is a parallel rewrite with the same output |
| 3-Models/betti | `baseline.py` | RF / XGBoost / MLP with default hyperparameters |
| 3-Models/betti | `tune.py` | Manual hyperparameter sweeps on the validation split (second and final round, see the note in the file) |
| 3-Models/betti | `tuned.py` | Test metrics with the tuned hyperparameters below |
| 2-Features/betti | `betti_minlife.py` | 64x64 features with short-lived persistence pairs removed (`MIN_LIFE` of 0, 5, 10, 20) |
| 3-Models/betti | `compare_minlife.py` | Validation AUC for the four `MIN_LIFE` settings |
| 2-Features/betti | `betti_size.py` | Same features at 128x128 and 224x224, original (leaky) splits |
| 3-Models/betti | `eval_size.py` | Test metrics at 128x128 / 224x224 on the leaky splits |
| 3-Models/betti | `eval_64.py` | Same eval on the 64x64 features, for comparison |
| 2-Features/betti | `betti_size_dedup.py` | Same as `betti_size.py` but reads `1-Data/splits/` |
| 3-Models/betti | `eval_size_dedup.py` | Test metrics at 128x128 / 224x224 on the current splits, same hyperparameters, not retuned |
| 3-Models/cnn | `check_torch.py`, `cnn.py` | Environment check, then the CNN baseline (reads `1-Data/splits/`) |
| 2-Features/radiomics | `extract.py` | pyradiomics features (first-order, GLCM, GLRLM, GLSZM, shape2D) on the current splits |
| 3-Models/radiomics | `xgb.py` | XGBoost train, tune and eval on those features |
| 2-Features/radiomics | `extract_texture.py` | First-order + texture only (adds GLDM and NGTDM, drops shape2D), 93 features |
| 3-Models/radiomics | `xgb_texture.py` | XGBoost on the texture-only features |
| 2-Features/cnn | `resnet18.py`, `resnet50.py` | Frozen ResNet embeddings (512-dim and 2048-dim) for the fusion models |
| 3-Models/fusion | `fusion18.py`, `fusion18_tuned.py`, `fusion50.py` | Feature-level fusion of Betti vectors and ResNet embeddings |
| 3-Models/fusion | `blend.py` | Prediction-level fusion (weighted blend and stacked logistic regression) |

To rebuild the gitignored files, run `betti_fast.py` (plus `betti_minlife.py` / `betti_size.py` for the variants), or `split_phash.py` then `betti_size_dedup.py` for the current splits.

## Betti classifiers

Betti vectors are standardized with `StandardScaler` (fit on train only) before going into any of these. The hyperparameters below were picked in `tune.py` on the (leaky) validation split and are used in every eval script, including `eval_size_dedup.py` on the new splits. Nothing was retuned after either leakage fix, so the before/after table above only shows the effect of the split changes.

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, min_samples_split=2, max_features="log2", class_weight="balanced"` |
| XGBoost | `n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8` |
| MLP | `hidden_layer_sizes=(128, 64), solver="adam", max_iter=500` |

All three use `random_state=42`.

Results on the current (phash-deduped) splits, from `eval_size_dedup.py`:

| Resolution | Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| 128x128 | Random Forest | 0.8541 | 0.9444 | 0.8375 | 0.7882 | 0.8933 | 0.8933 | 0.8256 |
| 128x128 | XGBoost | 0.8754 | 0.9509 | 0.8590 | 0.8204 | 0.9013 | 0.9013 | 0.8566 |
| 128x128 | MLP | 0.8765 | 0.9438 | 0.8571 | 0.8354 | 0.8800 | 0.8800 | 0.8740 |
| 224x224 | Random Forest | 0.8653 | 0.9496 | 0.8504 | 0.7986 | 0.9093 | 0.9093 | 0.8333 |
| 224x224 | XGBoost | 0.8732 | 0.9515 | 0.8571 | 0.8149 | 0.9040 | 0.9040 | 0.8508 |
| 224x224 | MLP | 0.8822 | 0.9498 | 0.8613 | 0.8534 | 0.8693 | 0.8693 | 0.8915 |

## CNN baseline (`3-Models/cnn/cnn.py`)

ResNet-18, ResNet-50, GoogLeNet and EfficientNet-B0, all ImageNet-pretrained and used as fixed feature extractors. Every parameter is frozen except the last layer (`fc` / `classifier`), which is replaced with a single linear output and trained for 3 epochs with Adam (`lr=1e-3`) and `BCEWithLogitsLoss`. Images are resized to 128x128 and normalized with the ImageNet mean and std. It runs on CPU because I don't have CUDA (see `check_torch.py`). There's no torch seed, so the numbers move a little between runs.

Results on the current splits, 128x128:

| Model | Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.8552 | 0.9390 | 0.8322 | 0.8122 | 0.8533 | 0.8533 | 0.8566 |
| ResNet-50 | 0.8810 | 0.9486 | 0.8490 | 0.9113 | 0.7947 | 0.7947 | 0.9438 |
| GoogLeNet | 0.8631 | 0.9345 | 0.8277 | 0.8799 | 0.7813 | 0.7813 | 0.9225 |
| EfficientNet-B0 | 0.8373 | 0.9182 | 0.8129 | 0.7875 | 0.8400 | 0.8400 | 0.8353 |

## Radiomics

`extract.py` runs pyradiomics on the same `1-Data/splits/` images: first-order, GLCM, GLRLM, GLSZM and 2D shape features (Mayerhoefer et al. set) at 224x224. That's 83 features, with 0 extraction failures across all 8,811 images.

There's no tumor segmentation mask in this dataset, so the ROI is basically the whole image. pyradiomics refuses a mask that's all foreground (it needs at least one background pixel to treat it as segmented), so I used the whole frame minus a 2px border. That gets past the check but it isn't a real region of interest. Radiomics is meant to be run on a segmented lesion. On the full frame the first-order and texture stats get diluted by background and normal anatomy, and the shape2D features are close to meaningless since every image has the same canvas and the same mask. As expected, no shape2D feature is in the top 15 importances.

pyradiomics has no prebuilt Windows wheel and needs a C++ compiler (Microsoft C++ Build Tools) to install. That's just setup, nothing in the code.

For XGBoost I started from the Betti hyperparameters (`n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6, colsample_bytree=0.8`), which gave a validation AUC of 0.9628 straight away. A one-at-a-time sweep found a slightly better combination (`max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0`), but it only moved the AUC from 0.9628 to 0.9630, which is noise. The Betti settings carried over fine to a feature space about 10x smaller.

Test results with `max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0, n_estimators=200`:

| Acc | AUC | F1 | Prec | Recall | Sens | Spec |
|---|---|---|---|---|---|---|
| 0.9080 | 0.9686 | 0.8932 | 0.8728 | 0.9147 | 0.9147 | 0.9031 |

Feature importance is mostly GLCM texture (`DifferenceAverage` 0.346 and `Idm` 0.210, over half of the total together), then more GLCM, GLRLM and first-order features.

0.9686 is in the same range as Betti (0.94 to 0.95) and the CNNs (0.92 to 0.95) on the same splits. I don't think it means radiomics is better, since all three have the same source problem described above. GLCM texture features are exactly what would pick up per-source differences like compression and contrast.

## Fusion

`3-Models/fusion/` has two kinds of fusion between the Betti vectors (128x128) and frozen ResNet embeddings. `fusion18.py` and `fusion50.py` project each side down to 32 dims, concatenate, and put a linear layer on top (the ResNet-18 one got an AUC of 0.9751 on test). `fusion18_tuned.py` adds hidden layers, BatchNorm, dropout and weight decay to that. `blend.py` combines the output probabilities of an XGBoost-on-Betti model and a ResNet-18 head, either with a blend weight picked on validation or a stacked logistic regression. All of it uses the same splits, so the source problem applies here too.

## Notes

- The 4 "channels" in the Betti vectors (grayscale mean, R, G, B) are mostly redundant. The X-rays are grayscale saved as RGB, so R, G and B are identical and 600 of the 800 dimensions repeat the grayscale channel.
- `tune.py` is only the second of two tuning rounds. I overwrote the first round's file before committing it, so those sweep values are gone.
