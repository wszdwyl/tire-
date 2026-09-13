# coding:utf-8
import re
import os
import cv2
import numpy as np
import logging
import main_model  # 导入自定义的模型结构模块，包含 TripletModel 类
import pandas as pd  # 用于保存结果到CSV

# --------------------------
# 配置部分
# --------------------------
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 图像目标大小 (高度, 宽度) 注意：根据模型输入调整
IMAGE_SIZE = (295, 295)  # 模型输入为 (160, 60, 3)

# 模型权重路径
MODEL_WEIGHTS_PATH = 'model_new.h5'

# 测试图像目录路径
TEST_PATH = './Data/test/test1/'

# 崭新轮胎（锚点）的文件名列表（5张）
ANCHOR_IMAGE_NAMES = ["new_1.jpg", "new_2.jpg", "new_3.jpg", "new_4.jpg", "new_5.jpg"]  # 替换为你的5张崭新轮胎文件名

# 磨损程度分类阈值（基于平均距离）
DISTANCE_THRESHOLDS = {
    "无磨损": 0.7,
    "轻微磨损": 1.0,
    "中度磨损": 1.2,
    "严重磨损": float('inf')  # 无穷大，表示大于8
}

# --------------------------
# 功能函数
# --------------------------

def get_id(filename: str) -> int:
    """
    从文件名中提取 ID。

    参数:
        filename (str): 文件名，假设文件名中包含数字。

    返回:
        int: 提取到的 ID。
    """
    match = re.search(r'\d+', os.path.splitext(filename)[0])
    if match:
        return int(match.group())
    else:
        logger.error(f"无法从文件名 '{filename}' 中提取 ID。")
        raise ValueError(f"无法从文件名 '{filename}' 中提取 ID。")

def load_and_preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """
    加载并预处理图像。

    参数:
        image_path (str): 图像文件的路径。
        target_size (tuple): 目标图像尺寸 (高度, 宽度)。

    返回:
        np.ndarray: 预处理后的图像数组，形状为 (1, 高度, 宽度, 3)。
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    # OpenCV 默认读取的图像是 (宽度, 高度)，需要转换为 (高度, 宽度)
    img = cv2.resize(img, dsize=(target_size[1], target_size[0]))  # 注意 dsize 是 (宽度, 高度)
    # 转换颜色空间从 BGR 到 RGB（如果模型需要 RGB）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 添加批次维度
    img = np.expand_dims(img, axis=0)
    return img

def classify_wear_level(avg_distance: float) -> str:
    """
    根据平均距离分类磨损程度。

    参数:
        avg_distance (float): 锚点与测试轮胎的平均距离。

    返回:
        str: 磨损程度类别（无磨损、轻微磨损、中度磨损、严重磨损）。
    """
    if avg_distance < DISTANCE_THRESHOLDS["无磨损"]:
        return "无磨损"
    elif avg_distance < DISTANCE_THRESHOLDS["轻微磨损"]:
        return "轻微磨损"
    elif avg_distance < DISTANCE_THRESHOLDS["中度磨损"]:
        return "中度磨损"
    else:
        return "严重磨损"

def get_true_wear_class(filename: str) -> str:
    """
    根据文件名获取真实磨损类别。

    参数:
        filename (str): 文件名。

    返回:
        str: 真实磨损类别（无磨损、轻微磨损、中度磨损、严重磨损）。
    """
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

# --------------------------
# 主评估函数
# --------------------------

def evaluate_model():
    """
    加载模型并对测试图像进行评估，计算准确率（多锚点平均版）。
    """
    # 加载模型
    model = main_model.TripletModel()
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        logger.info("模型权重加载成功。")
    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        raise

    # 获取测试目录下所有的 .jpg 文件
    try:
        test_images = [x for x in os.listdir(TEST_PATH) if x.lower().endswith('.jpg')]
    except Exception as e:
        logger.error(f"读取测试目录失败: {e}")
        raise

    n = len(test_images)
    if n < 2:
        logger.error("测试图像数量不足，至少需要两张图片进行比较。")
        return

    # --------------------------
    # 加载多张锚点图片并计算它们的嵌入特征
    # --------------------------
    anchor_embeddings = []  # 存储所有锚点的嵌入特征

    for anchor_name in ANCHOR_IMAGE_NAMES:
        anchor_path = os.path.join(TEST_PATH, anchor_name)
        try:
            anchor_img = load_and_preprocess_image(anchor_path, IMAGE_SIZE)
            # 模型预测：输入为 (锚点, 锚点, 锚点)
            predict = model.predict([anchor_img, anchor_img, anchor_img], verbose=0)
            predict = predict.ravel()
            # 假设模型输出为 3*128 维向量，前128维为锚点，中间128维为正样本，后128维为负样本
            anc_emb = predict[:128]  # 锚点部分的128维特征
            anchor_embeddings.append(anc_emb)
        except Exception as e:
            logger.error(f"加载锚点图片 {anchor_name} 失败: {e}")
            continue

    if not anchor_embeddings:
        logger.error("未成功加载任何锚点图片！")
        return

    # 计算锚点特征的平均值（作为无磨损的基准）
    avg_anchor_embedding = np.mean(anchor_embeddings, axis=0)

    # --------------------------
    # 遍历其他轮胎，逐个与平均锚点比较
    # 但由于模型只能接受图像输入，无法直接输入特征向量进行预测，
    # 因此需要改为：用多张锚点图片分别与测试轮胎比较，计算多个距离后取平均。
    # --------------------------
    correct = 0
    total = 0
    results = []

    for other_img_name in test_images:
        if other_img_name in ANCHOR_IMAGE_NAMES:
            continue  # 跳过锚点本身

        # 加载测试图片
        other_img_path = os.path.join(TEST_PATH, other_img_name)
        try:
            other_img = load_and_preprocess_image(other_img_path, IMAGE_SIZE)
        except Exception as e:
            logger.warning(f"跳过图片 {other_img_name} 由于加载失败: {e}")
            continue

        total += 1
        distances_for_this_img = []  # 存储该测试图片与所有锚点的距离

        # 用每个锚点分别与测试轮胎比较
        for anchor_name in ANCHOR_IMAGE_NAMES:
            anchor_path = os.path.join(TEST_PATH, anchor_name)
            try:
                anchor_img = load_and_preprocess_image(anchor_path, IMAGE_SIZE)
            except Exception as e:
                logger.error(f"加载锚点图片 {anchor_name} 失败: {e}")
                continue

            # 模型预测：输入为 (锚点, 测试轮胎, 测试轮胎)
            try:
               predict = model.predict([anchor_img, other_img, other_img], verbose=0)
            except Exception as e:
                logger.error(f"模型预测失败: {e}")
                continue

            predict = predict.ravel()
            if predict.shape[0] != 3 * 128:
                logger.error(f"预测结果的维度不正确，预期 {3*128}，实际 {predict.shape[0]}")
                continue

            anc_emb = predict[:128]  # 锚点部分的128维特征
            pos_emb = predict[128:256]  # 正样本（测试轮胎）部分的128维特征

            # 计算欧式距离
            dist = np.sum(np.square(anc_emb - pos_emb))
            distances_for_this_img.append(dist)

        if not distances_for_this_img:
            logger.warning(f"图片 {other_img_name} 未与任何锚点成功比较，跳过。")
            continue

        # 计算平均距离
        avg_distance = np.mean(distances_for_this_img)

        # 分类磨损程度
        predicted_class = classify_wear_level(avg_distance)

        # 获取真实类别
        true_class = get_true_wear_class(other_img_name)

        # 判断是否正确
        if predicted_class == true_class:
            correct += 1
            logger.info(f"{other_img_name}: 预测={predicted_class} (正确), 平均距离={avg_distance:.4f}")
        else:
            logger.info(f"{other_img_name}: 预测={predicted_class} (错误, 真实={true_class}), 平均距离={avg_distance:.4f}")

        # 保存结果
        results.append({
            "filename": other_img_name,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "avg_distance": avg_distance
        })

    # --------------------------
    # 计算正确率
    # --------------------------
    if total > 0:
        accuracy = (correct / total) * 100
        logger.info(f"评估完成。正确分类次数: {correct}, 总比较次数: {total}, 准确率: {accuracy:.2f}%")
    else:
        logger.warning("没有进行任何比较，无法计算准确率。")

    # 保存结果到CSV（可选）
    if results:
        try:
            df = pd.DataFrame(results)
            df.to_csv("test_results.csv", index=False)
            logger.info("结果已保存到 test_results.csv")
        except Exception as e:
            logger.error(f"保存结果到CSV失败: {e}")

if __name__ == "__main__":
    evaluate_model()