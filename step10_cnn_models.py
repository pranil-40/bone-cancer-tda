# step10: CNN baselines for comparison against the Betti-vector classifiers.
# Frozen-backbone transfer learning (only the final classifier layer is trained)
# on 4 ImageNet-pretrained architectures, CPU-only, images resized to 64x64.

import torch, torch.nn as nn, pandas as pd, numpy as np, os, time
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

base = r"C:\projects\bone_cancer_tda"
tfm = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class BoneDS(Dataset):
    """Loads an image split (train/test) straight from its _classes.csv, applying tfm on the fly."""
    def __init__(self, split):
        df = pd.read_csv(os.path.join(base, split, "_classes.csv")); df.columns = df.columns.str.strip()
        self.df, self.split = df, split
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(base, self.split, r["filename"])).convert("RGB")
        return tfm(img), int(r["cancer"])

train_dl = DataLoader(BoneDS("train"), batch_size=32, shuffle=True)
test_dl  = DataLoader(BoneDS("test"), batch_size=32)

def build_model(name):
    """Load a pretrained backbone, swap in a 1-output classifier head, and freeze everything else."""
    if name == "resnet18": m = models.resnet18(weights="DEFAULT"); m.fc = nn.Linear(m.fc.in_features, 1)
    if name == "resnet50": m = models.resnet50(weights="DEFAULT"); m.fc = nn.Linear(m.fc.in_features, 1)
    if name == "googlenet": m = models.googlenet(weights="DEFAULT", aux_logits=True); m.fc = nn.Linear(m.fc.in_features, 1)
    if name == "efficientnet": m = models.efficientnet_b0(weights="DEFAULT"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    for p in m.parameters(): p.requires_grad = False
    for p in (m.fc if hasattr(m,"fc") else m.classifier).parameters(): p.requires_grad = True
    return m

def train_eval(name, epochs=3):
    """Train the classifier head for a few epochs, then report test-set metrics (unfrozen params only)."""
    m = build_model(name)
    opt = torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    m.train()
    for ep in range(epochs):
        start = time.time()
        for x, y in train_dl:
            opt.zero_grad()
            out = m(x)
            out = out[0] if isinstance(out, tuple) else out
            loss = lossf(out.squeeze(1), y.float())
            loss.backward(); opt.step()
        print(f"  {name} epoch {ep+1}/{epochs} done in {time.time()-start:.0f}s")
    m.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            p = torch.sigmoid(m(x)).squeeze(1)
            probs += p.tolist(); labels += y.tolist()
    preds = [1 if p > 0.5 else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"{name}: Acc={accuracy_score(labels,preds):.4f} AUC={roc_auc_score(labels,probs):.4f} "
          f"F1={f1_score(labels,preds):.4f} Prec={precision_score(labels,preds):.4f} "
          f"Rec={recall_score(labels,preds):.4f} Sens={tp/(tp+fn):.4f} Spec={tn/(tn+fp):.4f}")

for name in ["resnet18", "resnet50", "googlenet", "efficientnet"]:
    train_eval(name)