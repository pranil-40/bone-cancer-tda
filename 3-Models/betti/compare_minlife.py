# Compares the MIN_LIFE = 0/5/10/20 feature sets from betti_minlife.py
# using a fixed Random Forest config, to see whether dropping short-lived
# persistence pairs helps validation AUC.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

for ml in [0, 5, 10, 20]:
    d = np.load(f"betti_64_minlife{ml}.npz")
    s = StandardScaler()
    Xtr, Xv = s.fit_transform(d["X_train"]), s.transform(d["X_val"])
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features="log2",
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, d["y_train"])
    auc = roc_auc_score(d["y_val"], rf.predict_proba(Xv)[:, 1])
    print(f"min_life={ml}: AUC={auc:.4f}")