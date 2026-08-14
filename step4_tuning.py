# step4: manual hyperparameter sweeps for RF/XGBoost/MLP on the step2 (64x64) features,
# scored on the validation split. Note: this file went through two rounds of tuning and
# the first round got overwritten on disk before it was ever committed, so what's below
# is the second (final) sweep only - the first round's exact ranges aren't recoverable.
# Winning values get carried into step5_best_models.py.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

data = np.load("betti_vectors_bone.npz")
X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
scaler = StandardScaler()
X_train_s, X_val_s = scaler.fit_transform(X_train), scaler.transform(X_val)

def score(model):
    """Fit on train, return validation AUC."""
    model.fit(X_train_s, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val_s)[:, 1])

def sweep(name, param, values, make_model):
    """Try each value for one hyperparameter and print the resulting validation AUC."""
    print(f"\n-- {param} --")
    for v in values:
        print(f"{param}={v}: AUC={score(make_model(v)):.4f}")

rf = lambda **kw: RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **kw)
xgb = lambda **kw: XGBClassifier(eval_metric="logloss", random_state=42, **kw)
mlp = lambda **kw: MLPClassifier(max_iter=500, random_state=42, **kw)

print("=== RANDOM FOREST ===")
sweep("RF", "n_estimators", [100, 200, 300, 500, 800], lambda v: rf(n_estimators=v, max_depth=20, max_features="log2"))
sweep("RF", "max_depth", [10, 15, 20, 30, None], lambda v: rf(n_estimators=200, max_depth=v, max_features="log2"))
sweep("RF", "min_samples_split", [2, 3, 5, 10], lambda v: rf(n_estimators=200, max_depth=20, min_samples_split=v, max_features="log2"))
sweep("RF", "max_features", ["sqrt", "log2", 0.3, 0.5, None], lambda v: rf(n_estimators=200, max_depth=20, max_features=v))

print("\n=== XGBOOST ===")
sweep("XGB", "learning_rate", [0.001, 0.01, 0.05, 0.1, 0.3, 0.5], lambda v: xgb(n_estimators=200, max_depth=12, learning_rate=v))
sweep("XGB", "max_depth", [4, 6, 9, 12, 15, 20], lambda v: xgb(n_estimators=200, max_depth=v, learning_rate=0.3))
sweep("XGB", "n_estimators", [50, 100, 200, 300, 500, 800], lambda v: xgb(n_estimators=v, max_depth=12, learning_rate=0.3))
sweep("XGB", "subsample", [0.4, 0.6, 0.8, 1.0], lambda v: xgb(n_estimators=200, max_depth=12, learning_rate=0.3, subsample=v))
sweep("XGB", "colsample_bytree", [0.4, 0.6, 0.8, 1.0], lambda v: xgb(n_estimators=200, max_depth=12, learning_rate=0.3, colsample_bytree=v))

print("\n=== MLP ===")
sweep("MLP", "hidden_layer_sizes", [(32,), (64,), (128,), (64,32), (128,64), (256,128,64)], lambda v: mlp(hidden_layer_sizes=v, solver="sgd"))
sweep("MLP", "solver", ["adam", "sgd", "lbfgs"], lambda v: mlp(hidden_layer_sizes=(64,), solver=v))