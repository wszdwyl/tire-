# coding: utf-8
"""Interpretable embedding baseline: four class prototypes instead of manual thresholds.

This replaces the old 'distance to new tires + hand-set thresholds' strategy.
It is recommended as a baseline/ablation comparator, not as the final proposed method.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

import main_model_v2

CLASS_NAMES = ["无磨损", "轻微磨损", "中度磨损", "严重磨损"]
CLASS_DIRS = [
    "./Data/train/new/", "./Data/train/slight/",
    "./Data/train/moderate/", "./Data/train/cracked/"
]
TEST_DIR = "./Data/test/test1/"
WEIGHTS = "encoder_best.weights.h5"


def files(path):
    p = Path(path)
    return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_from_name(path):
    n = os.path.basename(path).lower()
    if "new" in n:
        return "无磨损"
    if "slight" in n:
        return "轻微磨损"
    if "moderate" in n:
        return "中度磨损"
    if "cracked" in n:
        return "严重磨损"
    raise ValueError(path)


def load_batch(paths):
    x = np.empty((len(paths), 295, 295, 3), np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (295, 295), interpolation=cv2.INTER_AREA)
        x[i] = img.astype(np.float32) / 255.0
    return x


def main():
    enc = main_model_v2.build_base_network()
    enc.load_weights(WEIGHTS)

    prototypes = []
    for d in CLASS_DIRS:
        ps = files(d)
        emb = enc.predict(load_batch(ps), verbose=0)
        proto = np.mean(emb, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
    prototypes = np.stack(prototypes)

    test_paths = [p for p in files(TEST_DIR) if any(k in os.path.basename(p).lower()
                  for k in ("new", "slight", "moderate", "cracked"))]
    emb = enc.predict(load_batch(test_paths), verbose=0)
    dist = ((emb[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    pred_idx = np.argmin(dist, axis=1)
    pred = np.array([CLASS_NAMES[i] for i in pred_idx])
    true = np.array([label_from_name(p) for p in test_paths])

    print(f"prototype accuracy: {accuracy_score(true, pred):.4f}")
    print(classification_report(true, pred, labels=CLASS_NAMES, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
