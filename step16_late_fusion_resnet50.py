# step16: same late-fusion architecture and training config as step14
# (Ishtiaque's spec), swapping the CNN branch from ResNet-18 (512-dim) to
# ResNet-50 (2048-dim, via step13b). Everything else - batch_size=4, LR=1e-3
# substituted for XGBoost's shrinkage rate, early stopping patience=12,
# LR-reduction patience=5, best-checkpoint restore - is identical to step14
# so the two are a fair comparison. Same splits_fixed/ as everything else;
# same standing source-confound caveat, not suppressed here.

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

base = r"C:\projects\bone_cancer_tda"
LR = 1e-3
MAX_EPOCHS = 50
BATCH_SIZE = 4
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 12

torch.manual_seed(42)


def load_split_features(split):
    """Load the 800-dim Betti features (positional order) and the 2048-dim
    ResNet-50 features (filename-keyed), joined by filename so a silent
    misalignment can't slip through - same pattern as step14."""
    betti = np.load("betti_vectors_bone_128_fixed.npz")
    suffix = {"train": "_train", "valid": "_val", "test": "_test"}[split]
    X_tda = betti[f"X{suffix}"]
    y_tda = betti[f"y{suffix}"]

    manifest = pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))
    assert len(manifest) == len(X_tda), f"{split}: manifest has {len(manifest)} rows, betti features have {len(X_tda)}"

    resnet = np.load(f"resnet50_features_{split}.npz", allow_pickle=True)
    X_cnn, y_cnn, resnet_filenames = resnet["X"], resnet["y"], resnet["filename"]
    assert list(resnet_filenames) == list(manifest["filename"]), \
        f"{split}: resnet50 feature order doesn't match splits_fixed/{split}.csv order"
    assert np.array_equal(y_tda, y_cnn), f"{split}: betti and resnet50 labels disagree after alignment check"

    return X_tda, X_cnn, y_tda


class LateFusionNet(nn.Module):
    """TDA branch (800->32) + CNN branch (2048->32) -> concat(64) -> single-logit classifier."""
    def __init__(self, cnn_dim):
        super().__init__()
        self.tda_branch = nn.Sequential(nn.Linear(800, 32), nn.ReLU())
        self.cnn_branch = nn.Sequential(nn.Linear(cnn_dim, 32), nn.ReLU())
        self.classifier = nn.Linear(64, 1)

    def forward(self, tda, cnn):
        t = self.tda_branch(tda)
        c = self.cnn_branch(cnn)
        fused = torch.cat([t, c], dim=1)
        return self.classifier(fused).squeeze(1)


def to_tensors(X_tda, X_cnn, y):
    return (torch.tensor(X_tda, dtype=torch.float32),
            torch.tensor(X_cnn, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


@torch.no_grad()
def predict_probs(model, X_tda, X_cnn):
    model.eval()
    return torch.sigmoid(model(X_tda, X_cnn)).numpy()


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


if __name__ == "__main__":
    Xtr_tda, Xtr_cnn, ytr = load_split_features("train")
    Xva_tda, Xva_cnn, yva = load_split_features("valid")
    Xte_tda, Xte_cnn, yte = load_split_features("test")
    print(f"Loaded train={len(ytr)} valid={len(yva)} test={len(yte)}  "
          f"(TDA dim={Xtr_tda.shape[1]}, CNN dim={Xtr_cnn.shape[1]})")

    tda_scaler, cnn_scaler = StandardScaler(), StandardScaler()
    Xtr_tda = tda_scaler.fit_transform(Xtr_tda)
    Xva_tda = tda_scaler.transform(Xva_tda)
    Xte_tda = tda_scaler.transform(Xte_tda)
    Xtr_cnn = cnn_scaler.fit_transform(Xtr_cnn)
    Xva_cnn = cnn_scaler.transform(Xva_cnn)
    Xte_cnn = cnn_scaler.transform(Xte_cnn)

    Xtr_tda_t, Xtr_cnn_t, ytr_t = to_tensors(Xtr_tda, Xtr_cnn, ytr)
    Xva_tda_t, Xva_cnn_t, yva_t = to_tensors(Xva_tda, Xva_cnn, yva)
    Xte_tda_t, Xte_cnn_t, yte_t = to_tensors(Xte_tda, Xte_cnn, yte)

    train_dl = DataLoader(TensorDataset(Xtr_tda_t, Xtr_cnn_t, ytr_t),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = LateFusionNet(cnn_dim=Xtr_cnn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)
    lossf = nn.BCEWithLogitsLoss()

    best_auc, best_epoch, best_state, epochs_no_improve = -1.0, 0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xt, xc, y in train_dl:
            opt.zero_grad()
            out = model(xt, xc)
            loss = lossf(out, y)
            loss.backward()
            opt.step()

        val_probs = predict_probs(model, Xva_tda_t, Xva_cnn_t)
        val_auc = roc_auc_score(yva, val_probs)
        current_lr = opt.param_groups[0]["lr"]
        scheduler.step(val_auc)

        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:2d}: val_auc={val_auc:.4f}  lr={current_lr:.2e}  "
              f"{'(best)' if improved else f'(no improvement {epochs_no_improve}/{EARLY_STOP_PATIENCE})'}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}: no val AUC improvement for {EARLY_STOP_PATIENCE} epochs.")
            break
    else:
        print(f"\nReached MAX_EPOCHS={MAX_EPOCHS} without triggering early stopping.")

    print(f"\nBest validation AUC: {best_auc:.4f} at epoch {best_epoch} "
          f"(final epoch run: {epoch}, {epoch - best_epoch} epochs after best)")

    model.load_state_dict(best_state)
    test_probs = predict_probs(model, Xte_tda_t, Xte_cnn_t)
    metrics = evaluate(yte, test_probs)

    print(f"\n=== Test-set results (TDA+ResNet50 late fusion, best checkpoint epoch {best_epoch}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nFeature dimension flow: TDA {Xtr_tda.shape[1]} -> 32, CNN {Xtr_cnn.shape[1]} -> 32, "
          f"concat -> {32+32}, classifier -> 1")

    print(f"\n=== Comparison ===")
    print(f"TDA+ResNet18 fusion (step14):     AUC 0.9751")
    print(f"ResNet-50 alone (step10, CNN):     AUC 0.9486")
    print(f"TDA+ResNet50 fusion (this run):    AUC {metrics['AUC']:.4f}")
