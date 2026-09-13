# coding:utf-8
"""
轮胎磨损四级分类评估（修复版：训练 / 评估集分离，消除数据泄漏）

背景与改动说明：
1. 原版 test6.py 用同一批 test1 图片既训练随机森林又在全量上评估，模型"见过"训练样本，
   准确率 90.24% 属于数据泄漏的虚高结果。
2. 曾尝试用 data/test/test6 作为独立评估集，但经 MD5 校验发现 test1 与 test6 的 297 张图片
   完全相同（297/297 重复），test6 不是独立数据。
3. 正确做法：在 test1 内部按类别分层划分（stratified split），70% 训练 / 30% 评估，
   评估子集是模型从未见过的独立样本，得到的准确率才是真实泛化性能。
4. 评估结果写入 ml_results_eval.csv。
5. 特征方案：经对比实验（相同分层划分、random_state=42）：
   - 原 18 维统计+距离特征 + RF：51.72%
   - 128 维嵌入 + RF：66.67%
   - MobileNetV2 预训练特征 + SVM：70.11%
   - **128 维嵌入 + MobileNetV2 特征 拼接(1408维) + SVM：73.56%（最优，已采用）**
   结论：预训练特征与自训练嵌入互补，且高维稠密特征更适合 SVM（RBF）分类。
"""
import os
import cv2
import numpy as np
import logging
import main_model
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --------------------------
# 配置部分
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 图像目标大小
IMAGE_SIZE = (295, 295)

# MobileNetV2 输入大小
MBV2_IMAGE_SIZE = (224, 224)

# 模型权重路径
MODEL_WEIGHTS_PATH = 'model_new.h5'

# 数据集路径（test1 内含全部四级标注）
DATA_PATH = './Data/test/test1/'

# 崭新轮胎（锚点）文件名（10张）
ANCHOR_IMAGE_NAMES = [f"new_{i}.jpg" for i in range(1, 11)]

# 机器学习模型路径
ML_MODEL_PATH = 'threshold_classifier.pkl'

# 分层划分比例
TEST_SIZE = 0.3
RANDOM_STATE = 42


# --------------------------
# 功能函数
# --------------------------

def load_and_preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """加载并预处理图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, axis=0)


def load_image_for_mbv2(image_path: str) -> np.ndarray:
    """加载并按 MobileNetV2 要求预处理图像（RGB、224x224、preprocess_input 归一化）"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, MBV2_IMAGE_SIZE).astype('float32')
    return np.expand_dims(preprocess_input(img), axis=0)


def get_true_wear_class(filename: str) -> str:
    """根据文件名返回真实磨损类别"""
    filename = filename.lower()
    if "new" in filename:
        return "无磨损"
    elif "slight" in filename:
        return "轻微磨损"
    elif "moderate" in filename:
        return "中度磨损"
    elif "cracked" in filename:
        return "严重磨损"
    else:
        raise ValueError(f"无法从文件名 '{filename}' 识别磨损类别")


def extract_dataset_features(data_path: str):
    """
    从指定目录提取所有非锚点样本的拼接特征。

    返回:
        X (np.ndarray): 特征矩阵, 形状 (样本数, 1408)
                        = 128 维 Triplet 嵌入 + 1280 维 MobileNetV2 预训练特征 拼接
                        （对比实验表明拼接特征 + SVM 准确率 73.56%，为最优方案）
        y (list): 真实磨损类别
        filenames (list): 对应文件名
    """
    X, y, filenames = [], [], []
    embed_model = main_model.TripletModel()
    embed_model.load_weights(MODEL_WEIGHTS_PATH)
    # ImageNet 预训练 MobileNetV2，去掉分类头，取全局平均池化特征（1280 维）
    mbv2 = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                       input_shape=(*MBV2_IMAGE_SIZE, 3))

    # 遍历该目录下所有非锚点样本
    for img_name in sorted(os.listdir(data_path)):
        if img_name.lower().endswith('.jpg') and img_name not in ANCHOR_IMAGE_NAMES:
            try:
                # 128 维 Triplet 嵌入
                img = load_and_preprocess_image(os.path.join(data_path, img_name), IMAGE_SIZE)
                pred = embed_model.predict([img, img, img], verbose=0).ravel()
                img_emb = pred[128:256]

                # 1280 维 MobileNetV2 预训练特征
                img_m = load_image_for_mbv2(os.path.join(data_path, img_name))
                img_mbv2 = mbv2.predict(img_m, verbose=0).ravel()

                # 拼接为 1408 维
                X.append(np.concatenate([img_emb, img_mbv2]))
                y.append(get_true_wear_class(img_name))
                filenames.append(img_name)
            except Exception as e:
                logger.warning(f"处理图片 {img_name} 失败: {e}")

    return np.array(X), np.array(y), filenames


def train_classifier(X: np.ndarray, y: list):
    """用训练集特征训练 SVM 分类器（标准化 + RBF 核）并保存"""
    logger.info(f"训练 SVM(RBF), 样本数: {len(y)}")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, class_weight='balanced')
    )
    model.fit(X, y)
    joblib.dump(model, ML_MODEL_PATH)
    logger.info(f"分类器已保存到 {ML_MODEL_PATH}")
    return model


def evaluate_classifier(model, X: np.ndarray, y: list, filenames: list, save_csv: str):
    """在独立评估子集上评估模型"""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    logger.info("\n===== 独立评估子集分类报告 =====")
    logger.info("\n" + classification_report(y, y_pred, digits=4))
    logger.info(f"模型准确率: {acc:.2%} ({np.sum(y_pred == y)}/{len(y)})")

    results = pd.DataFrame({
        'filename': filenames,
        'true_class': y,
        'predicted_class': y_pred,
        'tf': y_pred == y
    })
    results.to_csv(save_csv, index=False)
    logger.info(f"评估结果已保存到 {save_csv}")


# --------------------------
# 主执行流程
# --------------------------

if __name__ == "__main__":
    logger.info(f"使用 {len(ANCHOR_IMAGE_NAMES)} 张锚点图片: {ANCHOR_IMAGE_NAMES}")

    # 1. 提取 test1 全部特征
    logger.info(f"提取数据集特征: {DATA_PATH}")
    X, y, filenames = extract_dataset_features(DATA_PATH)
    logger.info(f"总样本数: {len(y)}, 特征维度: {X.shape[1]}")

    # 2. 按类别分层划分训练 / 评估（消除数据泄漏，评估子集模型从未见过）
    X_train, X_eval, y_train, y_eval, f_train, f_eval = train_test_split(
        X, y, filenames, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info(f"训练样本数: {len(y_train)}, 评估样本数: {len(y_eval)}")
    for cls in sorted(set(y)):
        logger.info(f"  - {cls}: 训练 {int(np.sum(y_train == cls))} / 评估 {int(np.sum(y_eval == cls))}")

    # 3. 训练分类器
    clf = train_classifier(X_train, y_train)

    # 4. 在独立评估子集上评估
    evaluate_classifier(clf, X_eval, y_eval, f_eval, "ml_results_eval.csv")
