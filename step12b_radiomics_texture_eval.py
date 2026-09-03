# step12b: trains/evaluates XGBoost on the step11b texture-only radiomics
# features. Starts from the config that step12 already found transferred
# well to the (shape-included) radiomics feature space - not the original
# Betti-vector config - since that's the closer, more relevant reference
# point now. Same one-at-a-time sweep-around-a-reference-point approach as
# step4/step12, re-tuning only where a sweep actually beats the reused
# baseline.

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import os
base = r"C:\projects\bone_cancer_tda"

RADIOMICS_XGB_PARAMS = dict(n_estimators=200, max_depth=9, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=1.0)


def load_split(split):
    df = pd.read_csv(os.path.join(base, f"radiomics_texture_features_{split}.csv"))
    y = df["cancer"].to_numpy()
    X = df.drop(columns=["filename", "cancer"]).to_numpy()
    feature_names = [c for c in df.columns if c not in ("filename", "cancer")]
    return X, y, feature_names


def make_xgb(**overrides):
    params = {**RADIOMICS_XGB_PARAMS, **overrides}
    return XGBClassifier(eval_metric="logloss", random_state=42, **params)


def val_auc(params, X_train, y_train, X_val, y_val):
    """Fit with the given params and return validation AUC."""
    m = make_xgb(**params)
    m.fit(X_train, y_train)
    return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])


def sweep_and_pick(param_name, values, base_params, X_train, y_train, X_val, y_val):
    """Same style as step4/step12: vary one hyperparameter around the reused
    config, holding the others fixed, and report AUC for each value."""
    print(f"\n-- {param_name} (baseline={base_params[param_name]}) --")
    best_v, best_auc = base_params[param_name], None
    for v in values:
        auc = val_auc({**base_params, param_name: v}, X_train, y_train, X_val, y_val)
        print(f"  {param_name}={v}: AUC={auc:.4f}")
        if best_auc is None or auc > best_auc:
            best_auc, best_v = auc, v
    return best_v, best_auc


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
    }


if __name__ == "__main__":
    X_train, y_train, feature_names = load_split("train")
    X_val, y_val, _ = load_split("valid")
    X_test, y_test, _ = load_split("test")
    print(f"Loaded {len(feature_names)} texture-only radiomics features. "
          f"train={len(y_train)} valid={len(y_val)} test={len(y_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    baseline_auc = val_auc(RADIOMICS_XGB_PARAMS, X_train_s, y_train, X_val_s, y_val)
    print(f"\n=== Reused (previous-round) radiomics XGBoost hyperparameters, texture-only features ===")
    print(f"{RADIOMICS_XGB_PARAMS}")
    print(f"Validation AUC with reused hyperparameters: {baseline_auc:.4f}")

    print("\n=== Sequential one-at-a-time sweep around the reused config ===")
    tuned = dict(RADIOMICS_XGB_PARAMS)
    sweeps = [
        ("learning_rate", [0.01, 0.05, 0.1, 0.3, 0.5]),
        ("max_depth", [3, 6, 9, 12, 15]),
        ("n_estimators", [100, 200, 300, 500]),
        ("subsample", [0.6, 0.8, 1.0]),
        ("colsample_bytree", [0.6, 0.8, 1.0]),
    ]
    best_per_param = {}
    for name, values in sweeps:
        best_v, best_auc = sweep_and_pick(name, values, RADIOMICS_XGB_PARAMS, X_train_s, y_train, X_val_s, y_val)
        best_per_param[name] = best_v
        if best_v != RADIOMICS_XGB_PARAMS[name]:
            tuned[name] = best_v

    tuned_auc = val_auc(tuned, X_train_s, y_train, X_val_s, y_val)
    print(f"\nBest value per sweep: {best_per_param}")
    print(f"Combined config: {tuned}")
    print(f"Validation AUC, reused baseline vs combined-best: {baseline_auc:.4f} vs {tuned_auc:.4f}")

    if tuned_auc > baseline_auc + 1e-4:
        final_params = tuned
        print("Retuning improved validation AUC - using the combined config for test evaluation.")
    else:
        final_params = RADIOMICS_XGB_PARAMS
        print("Retuning did not improve on the reused hyperparameters - keeping them as-is.")

    print(f"\n=== Final XGBoost config: {final_params} ===")
    model = make_xgb(**final_params)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    m = evaluate(y_test, y_pred, y_prob)

    print("\n=== Test-set results (texture-only radiomics features, XGBoost) ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    top_idx = np.argsort(model.feature_importances_)[::-1][:15]
    print("\nTop 15 features by XGBoost importance:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {model.feature_importances_[i]:.4f}")
