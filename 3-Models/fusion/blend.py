# prediction-level fusion (fusion18.py fuses the features instead). two models
# each give a probability per image and I combine them two ways:
#   A. blend: w*p_tda + (1-w)*p_resnet, with w picked on validation AUC
#   B. stacking: logistic regression on [p_tda, p_resnet], fit on validation
#
# the TDA model reuses the tuned XGBoost params from eval_size.py. the ResNet-18
# model is the fc head from cnn.py (Adam 1e-3, 3 epochs, batch 32) trained on
# the saved features from resnet18.py instead of rerunning the CNN. one small
# difference: cnn.py ran with the backbone in train() mode so BatchNorm stats
# kept updating, resnet18.py uses eval(). shouldn't matter since the backbone
# isn't fine-tuned.

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
    print(f"\n{title}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# tda + xgboost branch

def load_betti(split):
    suffix = {"train": "_train", "valid": "_val", "test": "_test"}[split]
    d = np.load("betti_128_dedup.npz")
    return d[f"X{suffix}"], d[f"y{suffix}"]


def tda_xgboost_branch():
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


# resnet branch

def load_resnet_features(split):
    d = np.load(f"resnet18_features_{split}.npz", allow_pickle=True)
    return d["X"], d["y"], d["filename"]


def resnet_branch():
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


# alignment check

def check_alignment(split, y_tda, y_resnet):
    # both branches should line up with the manifest row order
    manifest = pd.read_csv(os.path.join(base, "1-Data", "splits", f"{split}.csv"))
    _, _, resnet_filenames = load_resnet_features(split)
    assert list(resnet_filenames) == list(manifest["filename"]), \
        f"{split}: resnet feature order doesn't match 1-Data/splits/{split}.csv order"
    assert len(y_tda) == len(manifest), f"{split}: betti row count doesn't match manifest"
    assert np.array_equal(y_tda, y_resnet), f"{split}: TDA and ResNet labels disagree after alignment check"


# fusion methods

def optimized_blend(p_tda_va, p_resnet_va, y_va, p_tda_te, p_resnet_te):
    # grid search w on validation AUC, then fix it for test
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
