# Same as extract_resnet18.py but with ResNet-50 (2048-dim embedding). Same
# reasoning: transfer_learning.py only trained the fc head, so the plain
# ImageNet backbone is equivalent. Feeds late_fusion_resnet50.py.

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

base = r"C:\projects\bone_cancer_tda"
SIZE = 128
tfm = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class BoneDS(Dataset):
    """Loads one split from 1-Data/splits/, applying tfm on the fly."""
    def __init__(self, split):
        self.df = pd.read_csv(os.path.join(base, "1-Data", "splits", f"{split}.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(base, r["source_split"], r["filename"])).convert("RGB")
        return tfm(img), int(r["cancer"]), r["filename"]


def build_backbone():
    """Plain ImageNet-pretrained ResNet-50 with its fc layer bypassed, so forward() returns the 2048-dim pooled embedding."""
    m = models.resnet50(weights="DEFAULT")
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def extract(split, backbone):
    """Run the frozen backbone over one split and collect its 2048-dim features, labels, and filenames."""
    ds = BoneDS(split)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    feats, labels, filenames = [], [], []
    start = time.time()
    with torch.no_grad():
        for x, y, fnames in dl:
            out = backbone(x)
            feats.append(out.numpy())
            labels.extend(y.tolist())
            filenames.extend(fnames)
    X = np.concatenate(feats, axis=0)
    print(f"{split}: {X.shape[0]} images -> {X.shape[1]}-dim features in {time.time()-start:.0f}s")
    return X, np.array(labels), np.array(filenames)


if __name__ == "__main__":
    backbone = build_backbone()
    for split in ["train", "valid", "test"]:
        X, y, filenames = extract(split, backbone)
        np.savez(f"resnet50_features_{split}.npz", X=X, y=y, filename=filenames)
        print(f"Saved resnet50_features_{split}.npz")
