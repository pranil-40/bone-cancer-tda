# step5: final test-set numbers for RF/XGBoost/MLP using the winning hyperparameters
# found in step4_tuning.py, on the original 64x64 Betti features.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

data = np.load("betti_vectors_bone.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=2,
                      max_features="log2", class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.3, subsample=0.6,
               colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), solver="adam", max_iter=500, random_state=42),
}

def evaluate(y_true, y_pred, y_prob):
    """Same metric bundle as step3_models.py, just written more compactly."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"Accuracy": accuracy_score(y_true, y_pred), "AUC": roc_auc_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred), "Sensitivity": tp/(tp+fn), "Specificity": tn/(tn+fp)}

print(f"{'Model':<15}{'Accuracy':<10}{'AUC':<10}{'F1':<10}{'Precision':<11}{'Recall':<10}{'Sensitivity':<13}{'Specificity':<12}")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    m = evaluate(y_test, y_pred, y_prob)
    print(f"{name:<15}" + "".join(f"{v:<10.4f}" for v in m.values()))