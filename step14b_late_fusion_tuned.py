# step14b: pushes step14's feature-level late-fusion model (already the best
# AUC in the project at 0.9751) as far as it will reasonably go. step14
# followed Ishtiaque's exact spec (single Linear(800->32)/Linear(512->32)
# branches, single-linear classifier head, batch_size=4) and is left
# untouched as the spec-compliant reference result. This is a deliberate
# departure from that spec, done for performance only:
#   - both branches get a hidden layer + BatchNorm + dropout instead of a
#     single linear projection, since 800-dim TDA input against ~7,000
#     train samples has real overfitting risk
#   - the fusion classifier becomes a small 2-layer MLP instead of one
#     linear layer, on the same reasoning
#   - Adam gets weight_decay (L2) as another overfitting lever
#   - dropout / weight_decay / batch_size / learning_rate are then swept
#     sequentially on validation AUC only, same one-at-a-time convention as
#     step4_tuning.py and step12_radiomics_eval.py - starting from the
#     architecture choices above as the baseline, not retuning those
# Same splits_fixed/ as everything else recently - still subject to the
# unresolved naming-family confound documented in the README.

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
MAX_EPOCHS = 50
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 12

BASELINE_CONFIG = {"dropout": 0.3, "weight_decay": 1e-4, "batch_size": 32, "lr": 1e-3}


def load_split_features(split):
    """Same load+join+alignment-check as step14."""
    betti = np.load("betti_vectors_bone_128_fixed.npz")
    suffix = {"train": "_train", "valid": "_val", "test": "_test"}[split]
    X_tda = betti[f"X{suffix}"]
    y_tda = betti[f"y{suffix}"]

    manifest = pd.read_csv(os.path.join(base, "splits_fixed", f"{split}.csv"))
    assert len(manifest) == len(X_tda), f"{split}: manifest has {len(manifest)} rows, betti features have {len(X_tda)}"

    resnet = np.load(f"resnet18_features_{split}.npz", allow_pickle=True)
    X_cnn, y_cnn, resnet_filenames = resnet["X"], resnet["y"], resnet["filename"]
    assert list(resnet_filenames) == list(manifest["filename"]), \
        f"{split}: resnet feature order doesn't match splits_fixed/{split}.csv order"
    assert np.array_equal(y_tda, y_cnn), f"{split}: betti and resnet labels disagree after alignment check"

    return X_tda, X_cnn, y_tda


class TunedLateFusionNet(nn.Module):
    """TDA branch (800->128->32) + CNN branch (512->64->32), each with
    BatchNorm+ReLU+Dropout, concat(64) -> small MLP classifier (64->32->1)."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.tda_branch = nn.Sequential(
            nn.Linear(800, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.cnn_branch = nn.Sequential(
            nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

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


def train(config, data, verbose=False):
    """Train one config to early-stop/max-epochs, return (best_val_auc, best_state)."""
    Xtr_tda_t, Xtr_cnn_t, ytr_t, Xva_tda_t, Xva_cnn_t, yva = data
    torch.manual_seed(42)

    train_dl = DataLoader(TensorDataset(Xtr_tda_t, Xtr_cnn_t, ytr_t),
                           batch_size=config["batch_size"], shuffle=True, drop_last=True)
    model = TunedLateFusionNet(dropout=config["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=LR_PATIENCE)
    lossf = nn.BCEWithLogitsLoss()

    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xt, xc, y in train_dl:
            opt.zero_grad()
            loss = lossf(model(xt, xc), y)
            loss.backward()
            opt.step()

        val_auc = roc_auc_score(yva, predict_probs(model, Xva_tda_t, Xva_cnn_t))
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc, best_state, epochs_no_improve = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"  epoch {epoch:2d}: val_auc={val_auc:.4f}  "
                  f"{'(best)' if epochs_no_improve == 0 else f'(no improvement {epochs_no_improve}/{EARLY_STOP_PATIENCE})'}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            break

    return best_auc, best_state


def sweep(name, param, values, base_config, data):
    print(f"\n-- {param} (baseline={base_config[param]}) --")
    best_val, best_auc = base_config[param], -1.0
    for v in values:
        cfg = dict(base_config, **{param: v})
        auc, _ = train(cfg, data)
        print(f"  {param}={v}: val_auc={auc:.4f}")
        if auc > best_auc:
            best_val, best_auc = v, auc
    return best_val


if __name__ == "__main__":
    Xtr_tda, Xtr_cnn, ytr = load_split_features("train")
    Xva_tda, Xva_cnn, yva = load_split_features("valid")
    Xte_tda, Xte_cnn, yte = load_split_features("test")
    print(f"Loaded train={len(ytr)} valid={len(yva)} test={len(yte)}")

    tda_scaler, cnn_scaler = StandardScaler(), StandardScaler()
    Xtr_tda = tda_scaler.fit_transform(Xtr_tda)
    Xva_tda = tda_scaler.transform(Xva_tda)
    Xte_tda = tda_scaler.transform(Xte_tda)
    Xtr_cnn = cnn_scaler.fit_transform(Xtr_cnn)
    Xva_cnn = cnn_scaler.transform(Xva_cnn)
    Xte_cnn = cnn_scaler.transform(Xte_cnn)

    Xtr_tda_t, Xtr_cnn_t, ytr_t = to_tensors(Xtr_tda, Xtr_cnn, ytr)
    Xva_tda_t, Xva_cnn_t, _ = to_tensors(Xva_tda, Xva_cnn, yva)
    Xte_tda_t, Xte_cnn_t, _ = to_tensors(Xte_tda, Xte_cnn, yte)
    data = (Xtr_tda_t, Xtr_cnn_t, ytr_t, Xva_tda_t, Xva_cnn_t, yva)

    baseline_auc, _ = train(BASELINE_CONFIG, data)
    print(f"\nBaseline (deeper arch, un-swept config {BASELINE_CONFIG}): val_auc={baseline_auc:.4f}")

    print("\n=== Sequential one-at-a-time sweep ===")
    config = dict(BASELINE_CONFIG)
    config["dropout"] = sweep("fusion", "dropout", [0.0, 0.1, 0.3, 0.5], config, data)
    config["weight_decay"] = sweep("fusion", "weight_decay", [0.0, 1e-5, 1e-4, 1e-3], config, data)
    config["batch_size"] = sweep("fusion", "batch_size", [4, 16, 32, 64], config, data)
    config["lr"] = sweep("fusion", "lr", [5e-4, 1e-3, 2e-3], config, data)
    print(f"\nCombined best config: {config}")

    final_auc, final_state = train(config, data, verbose=True)
    print(f"\nBest validation AUC with combined config: {final_auc:.4f} "
          f"(baseline was {baseline_auc:.4f})")

    model = TunedLateFusionNet(dropout=config["dropout"])
    model.load_state_dict(final_state)
    test_probs = predict_probs(model, Xte_tda_t, Xte_cnn_t)
    metrics = evaluate(yte, test_probs)

    print(f"\n=== Test-set results (tuned late fusion, config={config}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
