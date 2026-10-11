# Test-set numbers for RF / XGBoost / MLP on the resolution-swept features from
# betti_resolution.py (128x128 or 224x224). Compare with eval_64.py and
# tuned_models.py for 64x64. These use the original leaky split, so read the
# data leakage section of the README before taking the comparison too seriously.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

SIZE = 224  # 128 or 224, match the betti_resolution.py run

d = np.load(f"betti_{SIZE}_orig.npz")
s = StandardScaler()
Xtr, Xte = s.fit_transform(d["X_train"]), s.transform(d["X_test"])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=20, max_features="log2",
                      class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6,
               colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), solver="adam", max_iter=500, random_state=42),
}

print(f"=== Results at {SIZE}x{SIZE} ===")
for name, m in models.items():
    m.fit(Xtr, d["y_train"])
    pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(d["y_test"], pred).ravel()
    print(f"{name}: Acc={accuracy_score(d['y_test'],pred):.4f} AUC={roc_auc_score(d['y_test'],prob):.4f} "
          f"F1={f1_score(d['y_test'],pred):.4f} Prec={precision_score(d['y_test'],pred):.4f} "
          f"Rec={recall_score(d['y_test'],pred):.4f} Sens={tp/(tp+fn):.4f} Spec={tn/(tn+fp):.4f}")