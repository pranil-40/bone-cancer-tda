# step15: prediction-level (late) fusion, as opposed to step14's feature-level
# fusion. Two independently-trained branches each produce a probability per
# image, and those probabilities are combined two ways:
#   Method A - optimized blend: p_final = w*p_tda + (1-w)*p_resnet, with w
#     grid-searched on validation AUC only, then applied fixed to test.
#   Method B - stacked: a 2-feature logistic regression meta-classifier
#     ([p_tda, p_resnet] -> label), trained on validation only, applied to test.
# Both branches and both fusion methods report on the same splits_fixed/
# (phash-deduped) split as everything else recently - still subject to the
# unresolved naming-family confound documented in the README.
#
# Branch 1 (TDA+XGBoost) reuses the already-tuned hyperparameters from
# step9/D.3 verbatim - not retuned.
# Branch 2 (ResNet-18) reconstructs the "frozen backbone + trained fc head"
# setup from step10_cnn_models.py's resnet18 run (Adam lr=1e-3,
# BCEWithLogitsLoss, 3 epochs, batch_size=32) but trains that head on
# step13's pre-extracted 512-dim frozen features rather than re-running the
# CNN forward pass every step - mathematically the same, since the backbone
# is frozen either way, except step10's forward passes ran with the backbone
# in .train() mode, which lets BatchNorm update its running stats even
# though its learnable params are frozen; step13 extracted in .eval() mode.
# That's a documented simplification, not expected to matter much for a
# backbone that was never fine-tuned in the first place.

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

base = r"C:\projects\bone_cancer_tda"
torch.manual_seed(42)


def evaluate(y_true, probs, threshold=0.5):
    """Same 7-metric report used throughout this project."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "AUC": roc_auc_score(y_true, probs),
        "F1": f1_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
    }


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# ---------------------------------------------------------------- Branch 1

def load_betti(split):
    suffix = {"train": "_train", "valid": "_val", "test": "_test"}[split]
    d = np.load("betti_vectors_bone_128_fixed.npz")
    return d[f"X{suffix}"], d[f"y{suffix}"]


def tda_xgboost_branch():
    """Train the already-tuned XGBoost model (step9/D.3 hyperparameters, not
    retuned) on the 800-dim Betti vectors, return probabilities on valid/test."""
    Xtr, ytr = load_betti("train")
    Xva, yva = load_betti("valid")
    Xte, yte = load_betti("test")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    model = XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.3,
                           subsample=0.6, colsample_bytree=0.8,
                           eval_metric="logloss", random_state=42)
    model.fit(Xtr, ytr)

    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    return p_va, yva, p_te, yte


# ---------------------------------------------------------------- Branch 2

def load_resnet_features(split):
    d = np.load(f"resnet18_features_{split}.npz", allow_pickle=True)
    return d["X"], d["y"], d["filename"]


def resnet_branch():
    """Reconstruct step10_cnn_models.py's resnet18 setup - frozen backbone
    (step13's cached 512-dim features), trained single-linear classifier head,
    same optimizer/loss/epoch count - and return probabilities on valid/test."""
    Xtr, ytr, _ = load_resnet_features("train")
    Xva, yva, _ = load_resnet_features("valid")
    Xte, yte, _ = load_resnet_features("test")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    head = nn.Linear(512, 1)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

    head.train()
    for epoch in range(3):
        for x, y in train_dl:
            opt.zero_grad()
            loss = lossf(head(x).squeeze(1), y)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(head(Xva_t).squeeze(1)).numpy()
        p_te = torch.sigmoid(head(Xte_t).squeeze(1)).numpy()
    return p_va, yva, p_te, yte


# ---------------------------------------------------------------- alignment

def check_alignment(split, y_tda, y_resnet):
    """Same filename-order sanity check used in step14 - confirms both branches'
    valid/test rows refer to the same images in the same order before fusing."""
    manifest = pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))
    _, _, resnet_filenames = load_resnet_features(split)
    assert list(resnet_filenames) == list(manifest["filename"]), \
        f"{split}: resnet feature order doesn't match splits_fixed/{split}.csv order"
    assert len(y_tda) == len(manifest), f"{split}: betti row count doesn't match manifest"
    assert np.array_equal(y_tda, y_resnet), f"{split}: TDA and ResNet labels disagree after alignment check"


# ---------------------------------------------------------------- fusion

def optimized_blend(p_tda_va, p_resnet_va, y_va, p_tda_te, p_resnet_te):
    """Method A: grid-search blend weight w on validation AUC only, apply fixed to test."""
    best_w, best_auc = None, -1.0
    for w in np.arange(0.0, 1.0001, 0.05):
        blend = w * p_tda_va + (1 - w) * p_resnet_va
        auc = roc_auc_score(y_va, blend)
        if auc > best_auc:
            best_w, best_auc = round(w, 2), auc
    print(f"\nOptimized blend: best w={best_w} (validation AUC={best_auc:.4f})")
    test_blend = best_w * p_tda_te + (1 - best_w) * p_resnet_te
    return test_blend, best_w


def stacked_meta_classifier(p_tda_va, p_resnet_va, y_va, p_tda_te, p_resnet_te):
    """Method B: 2-feature logistic regression meta-classifier, trained on
    validation [p_tda, p_resnet] pairs only, applied to test."""
    Xva = np.column_stack([p_tda_va, p_resnet_va])
    Xte = np.column_stack([p_tda_te, p_resnet_te])
    meta = LogisticRegression(random_state=42)
    meta.fit(Xva, y_va)
    print(f"\nStacked meta-classifier coefficients: "
          f"w_tda={meta.coef_[0][0]:.4f}, w_resnet={meta.coef_[0][1]:.4f}, "
          f"intercept={meta.intercept_[0]:.4f}")
    test_probs = meta.predict_proba(Xte)[:, 1]
    return test_probs


if __name__ == "__main__":
    print("Training TDA+XGBoost branch...")
    p_tda_va, y_va_tda, p_tda_te, y_te_tda = tda_xgboost_branch()

    print("Training ResNet-18 branch...")
    p_resnet_va, y_va_resnet, p_resnet_te, y_te_resnet = resnet_branch()

    check_alignment("valid", y_va_tda, y_va_resnet)
    check_alignment("test", y_te_tda, y_te_resnet)
    y_va, y_te = y_va_tda, y_te_tda

    print_metrics("1. TDA+XGBoost branch alone (test)", evaluate(y_te, p_tda_te))
    print_metrics("2. ResNet-18 branch alone (test)", evaluate(y_te, p_resnet_te))

    blend_probs, best_w = optimized_blend(p_tda_va, p_resnet_va, y_va, p_tda_te, p_resnet_te)
    print_metrics(f"3. Optimized blend, w={best_w} (test)", evaluate(y_te, blend_probs))

    stacked_probs = stacked_meta_classifier(p_tda_va, p_resnet_va, y_va, p_tda_te, p_resnet_te)
    print_metrics("4. Stacked meta-classifier (test)", evaluate(y_te, stacked_probs))
