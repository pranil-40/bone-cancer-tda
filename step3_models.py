# step3: baseline classifiers on the step2 Betti vectors, default hyperparameters.
# this is the "does the feature representation carry any signal at all" check
# before spending time tuning anything (see step4_tuning.py).

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier

data = np.load("betti_vectors_bone.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False,
                             eval_metric="logloss", random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}

def evaluate(y_true, y_pred, y_prob):
    """Bundle the usual classification metrics plus sensitivity/specificity into one dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,

    
    }

print(f"{'Model':<15}{'Accuracy':<10}{'AUC':<10}{'F1':<10}{'Precision':<11}{'Recall':<10}{'Sensitivity':<13}{'Specificity':<12}")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    m = evaluate(y_test, y_pred, y_prob)
    print(f"{name:<15}" + "".join(f"{m[k]:<10.4f}" if k not in ("Precision","Sensitivity","Specificity")
          else f"{m[k]:<11.4f}" for k in m))