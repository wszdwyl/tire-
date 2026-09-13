# coding: utf-8
"""Independent four-grade tire-wear evaluation + ablation study.

Proposed feature fusion:
- 128-D ordinal triplet embedding (self-trained tire feature)
- 1280-D ImageNet MobileNetV2 feature (transfer feature)
- concatenation -> standardized RBF-SVM

The script also evaluates each branch separately to produce an ablation table.
It checks exact MD5 duplicates between training and test sets and removes duplicate
items from the test set to prevent leakage.
"""

import hashlib
import logging
import os
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import main_model_v2


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["无磨损", "轻微磨损", "中度磨损", "严重磨损"]
CLASS_DIRS = [
    "./Data/train/new/",
    "./Data/train/slight/",
    "./Data/train/moderate/",
    "./Data/train/cracked/",
]
ANCHOR_DIR = "./Data/anchor/"
TEST_DIR = "./Data/test/test1/"
ENCODER_WEIGHTS = "encoder_best.weights.h5"
BATCH_SIZE = 16
RANDOM_STATE = 42


def _img_files(path):
    p = Path(path)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})


def get_true_wear_class(filename):
    name = os.path.basename(filename).lower()
    if "new" in name:
        return "无磨损"
    if "slight" in name:
        return "轻微磨损"
    if "moderate" in name:
        return "中度磨损"
    if "cracked" in name:
        return "严重磨损"
    raise ValueError(f"无法由文件名识别类别: {filename}")


def collect_train_set():
    paths, labels = [], []
    for label, d in zip(CLASS_NAMES, CLASS_DIRS):
        xs = _img_files(d)
        paths.extend(xs)
        labels.extend([label] * len(xs))
    xs = _img_files(ANCHOR_DIR)
    paths.extend(xs)
    labels.extend(["无磨损"] * len(xs))
    return paths, np.array(labels)


def collect_test_set():
    paths, labels = [], []
    for p in _img_files(TEST_DIR):
        try:
            labels.append(get_true_wear_class(p))
            paths.append(p)
        except ValueError:
            logger.warning("跳过无法识别标签的测试图像: %s", p)
    return paths, np.array(labels)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def remove_exact_train_test_duplicates(train_paths, test_paths, test_labels):
    train_hashes = {md5(p) for p in train_paths}
    keep_paths, keep_labels, removed = [], [], []
    for p, y in zip(test_paths, test_labels):
        if md5(p) in train_hashes:
            removed.append(p)
        else:
            keep_paths.append(p)
            keep_labels.append(y)
    if removed:
        logger.warning("发现并移除 %d 张训练/测试完全重复图像，防止数据泄漏。", len(removed))
        pd.DataFrame({"removed_duplicate": removed}).to_csv(
            "removed_test_duplicates.csv", index=False, encoding="utf-8-sig"
        )
    return keep_paths, np.array(keep_labels)


def load_encoder_batch(paths):
    arr = np.empty((len(paths), 295, 295, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (295, 295), interpolation=cv2.INTER_AREA)
        arr[i] = img.astype(np.float32) / 255.0
    return arr


def load_mbv2_batch(paths):
    arr = np.empty((len(paths), 224, 224, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        arr[i] = preprocess_input(img)
    return arr


def batched_predict(model, paths, loader):
    feats = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        feats.append(model.predict(loader(batch_paths), verbose=0))
    return np.concatenate(feats, axis=0)


def extract_features(train_paths, test_paths):
    encoder = main_model_v2.build_base_network()
    encoder.load_weights(ENCODER_WEIGHTS)

    mbv2 = MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    mbv2.trainable = False

    logger.info("提取 128-D 轮胎度量嵌入...")
    tr_trip = batched_predict(encoder, train_paths, load_encoder_batch)
    te_trip = batched_predict(encoder, test_paths, load_encoder_batch)

    logger.info("提取 1280-D MobileNetV2 迁移特征...")
    tr_mob = batched_predict(mbv2, train_paths, load_mbv2_batch)
    te_mob = batched_predict(mbv2, test_paths, load_mbv2_batch)

    return {
        "Triplet-128": (tr_trip, te_trip),
        "MobileNetV2-1280": (tr_mob, te_mob),
        "Fusion-1408": (
            np.concatenate([tr_trip, tr_mob], axis=1),
            np.concatenate([te_trip, te_mob], axis=1),
        ),
    }


def build_tuned_svm(X, y):
    counts = Counter(y)
    min_count = min(counts.values())
    n_splits = min(5, min_count)
    if n_splits < 2:
        raise ValueError(f"每类训练样本至少需要2张，当前类别计数: {dict(counts)}")

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced")),
    ])
    params = {
        "svc__C": [1, 10, 50],
        "svc__gamma": ["scale", 1e-3, 1e-2],
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, params, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search


def metrics_row(mode, y_true, y_pred, best_params, cv_score):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0
    )
    idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    ordinal_mae = float(np.mean([abs(idx[a] - idx[b]) for a, b in zip(y_true, y_pred)]))
    return {
        "feature_mode": mode,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "quadratic_kappa": cohen_kappa_score(
            [idx[x] for x in y_true], [idx[x] for x in y_pred], weights="quadratic"
        ),
        "ordinal_mae": ordinal_mae,
        "train_cv_macro_f1": cv_score,
        "best_params": str(best_params),
    }


def main():
    train_paths, y_train = collect_train_set()
    test_paths, y_test = collect_test_set()
    logger.info("训练样本: %d, 测试样本: %d", len(train_paths), len(test_paths))
    logger.info("训练类别: %s", dict(Counter(y_train)))
    logger.info("测试类别: %s", dict(Counter(y_test)))

    if not train_paths or not test_paths:
        raise FileNotFoundError("训练集或测试集为空，请检查路径。")

    test_paths, y_test = remove_exact_train_test_duplicates(train_paths, test_paths, y_test)
    if not test_paths:
        raise RuntimeError("去除重复图像后测试集为空，必须准备独立测试集。")

    feature_sets = extract_features(train_paths, test_paths)
    rows = []

    for mode, (X_train, X_test) in feature_sets.items():
        logger.info("===== %s =====", mode)
        search = build_tuned_svm(X_train, y_train)
        pred = search.predict(X_test)

        row = metrics_row(mode, y_test, pred, search.best_params_, search.best_score_)
        rows.append(row)
        logger.info("test accuracy=%.4f, macro-F1=%.4f, ordinal-MAE=%.4f",
                    row["accuracy"], row["macro_f1"], row["ordinal_mae"])
        logger.info("\n%s", classification_report(
            y_test, pred, labels=CLASS_NAMES, digits=4, zero_division=0
        ))

        cm = confusion_matrix(y_test, pred, labels=CLASS_NAMES)
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            f"confusion_{mode}.csv", encoding="utf-8-sig"
        )
        pd.DataFrame({
            "filename": [os.path.basename(p) for p in test_paths],
            "true_class": y_test,
            "predicted_class": pred,
            "correct": pred == y_test,
        }).to_csv(f"predictions_{mode}.csv", index=False, encoding="utf-8-sig")

        if mode == "Fusion-1408":
            joblib.dump(search.best_estimator_, "fusion_svm.pkl")

    result_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result_df.to_csv("ablation_results.csv", index=False, encoding="utf-8-sig")
    logger.info("\n===== 消融实验汇总 =====\n%s", result_df.to_string(index=False))


if __name__ == "__main__":
    main()
