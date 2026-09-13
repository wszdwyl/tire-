# coding: utf-8
"""Ordinal-aware triplet data loader for four-level tire wear.

Classes are ordered as:
0 no wear -> 1 slight -> 2 moderate -> 3 severe.

Training strategy:
- file-level train/validation split to avoid image leakage;
- class-balanced anchor sampling;
- hard negatives are preferentially sampled from adjacent wear levels;
- the triplet margin increases with ordinal class distance;
- augmentation is performed online only for training data.
"""

import math
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 295
CLASS_NAMES = ["new", "slight", "moderate", "cracked"]
CLASS_DIRS = [
    "./Data/train/new/",
    "./Data/train/slight/",
    "./Data/train/moderate/",
    "./Data/train/cracked/",
]
ANCHOR_DIR = "./Data/anchor/"


def _jpgs(path):
    p = Path(path)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})


def collect_class_paths(include_anchor_in_new=True):
    """Collect four-class image paths. Extra new-tire anchors can join class 0."""
    class_paths = [_jpgs(p) for p in CLASS_DIRS]
    if include_anchor_in_new:
        class_paths[0] = class_paths[0] + _jpgs(ANCHOR_DIR)

    for name, paths in zip(CLASS_NAMES, class_paths):
        if len(paths) < 2:
            raise FileNotFoundError(f"类别 {name} 可用图像少于2张，请检查目录。当前={len(paths)}")
    return class_paths


def stratified_file_split(class_paths, val_ratio=0.2, seed=42):
    """Split each class at file level, preventing the same file entering train and validation."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for paths in class_paths:
        paths = np.array(paths, dtype=object)
        idx = rng.permutation(len(paths))
        n_val = max(1, int(round(len(paths) * val_ratio)))
        n_val = min(n_val, len(paths) - 1)
        val.append(paths[idx[:n_val]].tolist())
        train.append(paths[idx[n_val:]].tolist())
    return train, val


def _load_rgb01(path, augment=False, rng=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    if augment:
        rng = rng or np.random.default_rng()
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)

        angle = float(rng.uniform(-10.0, 10.0))
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        alpha = float(rng.uniform(0.90, 1.10))
        beta = float(rng.uniform(-10.0, 10.0))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img.astype(np.float32) / 255.0


class OrdinalTripletSequence(tf.keras.utils.Sequence):
    """Disk-based triplet sequence to avoid loading all 295x295 images into RAM."""

    def __init__(
        self,
        class_paths,
        triplets_per_epoch=2400,
        batch_size=12,
        hard_negative_prob=0.70,
        base_margin=0.25,
        margin_step=0.15,
        augment=True,
        seed=42,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_paths = class_paths
        self.triplets_per_epoch = int(triplets_per_epoch)
        self.batch_size = int(batch_size)
        self.hard_negative_prob = float(hard_negative_prob)
        self.base_margin = float(base_margin)
        self.margin_step = float(margin_step)
        self.augment = bool(augment)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self._triplets = []
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.triplets_per_epoch / self.batch_size)

    def _sample_triplet(self, anchor_class):
        a_pool = self.class_paths[anchor_class]
        a_idx = int(self.rng.integers(0, len(a_pool)))
        p_idx = int(self.rng.integers(0, len(a_pool) - 1))
        if p_idx >= a_idx:
            p_idx += 1

        adjacent = [c for c in (anchor_class - 1, anchor_class + 1) if 0 <= c < 4]
        if adjacent and self.rng.random() < self.hard_negative_prob:
            neg_class = int(self.rng.choice(adjacent))
        else:
            neg_candidates = [c for c in range(4) if c != anchor_class]
            neg_class = int(self.rng.choice(neg_candidates))

        n_pool = self.class_paths[neg_class]
        n_path = n_pool[int(self.rng.integers(0, len(n_pool)))]
        class_gap = abs(anchor_class - neg_class)
        margin = self.base_margin + self.margin_step * (class_gap - 1)
        return a_pool[a_idx], a_pool[p_idx], n_path, np.float32(margin)

    def on_epoch_end(self):
        desc = [self._sample_triplet(i % 4) for i in range(self.triplets_per_epoch)]
        if self.shuffle:
            self.rng.shuffle(desc)
        self._triplets = desc

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.triplets_per_epoch)
        batch = self._triplets[start:end]
        b = len(batch)

        anc = np.empty((b, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        pos = np.empty_like(anc)
        neg = np.empty_like(anc)
        margins = np.empty((b, 1), dtype=np.float32)

        for i, (a_path, p_path, n_path, margin) in enumerate(batch):
            anc[i] = _load_rgb01(a_path, augment=self.augment, rng=self.rng)
            pos[i] = _load_rgb01(p_path, augment=self.augment, rng=self.rng)
            neg[i] = _load_rgb01(n_path, augment=self.augment, rng=self.rng)
            margins[i, 0] = margin

        return (anc, pos, neg), margins


def build_sequences(
    val_ratio=0.2,
    batch_size=12,
    train_triplets=2400,
    val_triplets=600,
    seed=42,
):
    class_paths = collect_class_paths(include_anchor_in_new=True)
    train_paths, val_paths = stratified_file_split(class_paths, val_ratio=val_ratio, seed=seed)

    train_seq = OrdinalTripletSequence(
        train_paths,
        triplets_per_epoch=train_triplets,
        batch_size=batch_size,
        augment=True,
        seed=seed,
        shuffle=True,
    )
    val_seq = OrdinalTripletSequence(
        val_paths,
        triplets_per_epoch=val_triplets,
        batch_size=batch_size,
        augment=False,
        seed=seed + 1,
        shuffle=False,
    )
    return train_seq, val_seq, train_paths, val_paths


if __name__ == "__main__":
    tr, va, tr_paths, va_paths = build_sequences()
    print("train class counts:", [len(x) for x in tr_paths])
    print("val class counts:", [len(x) for x in va_paths])
    (a, p, n), m = tr[0]
    print("batch:", a.shape, p.shape, n.shape, "margins:", m[:8].ravel())
