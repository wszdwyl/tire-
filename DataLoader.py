import cv2  # 导入 OpenCV 库，用于图像处理
import numpy as np  # 导入 NumPy 库，用于数值计算
import os  # 导入 os 库，用于文件和目录操作

# ============================================================
# 数据目录（四级磨损分类版）
#   anchor/           -> 锚点：崭新轮胎（无磨损）基准
#   train/new/        -> 无磨损
#   train/slight/     -> 轻微磨损
#   train/moderate/   -> 中度磨损
#   train/cracked/    -> 严重磨损
# ============================================================
anchor_path = "./Data/anchor/"  # 锚点图像文件夹路径
CLASS_DIRS = [  # 四级磨损类别目录（顺序即类别顺序）
    "./Data/train/new/",        # 无磨损
    "./Data/train/slight/",     # 轻微磨损
    "./Data/train/moderate/",   # 中度磨损
    "./Data/train/cracked/",    # 严重磨损
]
IMG_SIZE = 295  # 图像统一尺寸


def _load_dir(path):
    """读取某目录下所有 .jpg 图片并统一 resize 为 (295,295,3)，返回 float32 数组"""
    imgs = [x for x in sorted(os.listdir(path)) if x.lower().endswith('.jpg')]
    if len(imgs) == 0:
        raise FileNotFoundError(f"目录为空或没有 .jpg 图片: {path}")
    data = np.empty((len(imgs), IMG_SIZE, IMG_SIZE, 3), dtype='float32')
    for i, name in enumerate(imgs):
        im = cv2.imread(os.path.join(path, name))
        if im is None:
            raise FileNotFoundError(f"无法读取图片: {os.path.join(path, name)}")
        im = cv2.resize(im, dsize=(IMG_SIZE, IMG_SIZE))
        data[i] = im
    return data


def dataloder():
    """
    四级三元组数据加载。

    返回:
        anc_data : 锚点（崭新轮胎基准），形状 (N,295,295,3)
        pos_data : 正样本（无磨损，与锚点同类），形状 (N,295,295,3)
        neg_data : 负样本（从 轻微/中度/严重 三级中随机采样），形状 (N,295,295,3)
        labels   : 标签占位数组，形状 (N,2)

    说明:
        - 三个输入数组第一维统一为 N（=锚点数量），保证 model.fit 可正常训练；
        - 负样本覆盖轻微/中度/严重三个磨损级别，让嵌入能学到更细的磨损差异；
        - 数量不足时使用有放回随机采样补齐。
    """
    # 加载锚点与四类样本
    anc_data = _load_dir(anchor_path)              # 崭新轮胎基准
    new_data = _load_dir(CLASS_DIRS[0])            # 无磨损
    wear_data = np.concatenate(                    # 轻微+中度+严重 合并为磨损负样本池
        [_load_dir(p) for p in CLASS_DIRS[1:]], axis=0
    )

    # 统一三元组数量为锚点数量 N
    n = len(anc_data)

    # 正样本：从无磨损类有放回采样 N 张
    pos_idx = np.random.randint(0, len(new_data), n)
    pos_data = new_data[pos_idx]

    # 负样本：从磨损类有放回采样 N 张
    neg_idx = np.random.randint(0, len(wear_data), n)
    neg_data = wear_data[neg_idx]

    # 标签占位（Triplet Loss 不使用标签，仅保持 fit 接口完整）
    labels = np.tile([1, 0], (n, 1))

    return anc_data, pos_data, neg_data, labels


if __name__ == "__main__":
    a, p, n_, l = dataloder()
    print(f"锚点: {a.shape}, 正样本: {p.shape}, 负样本: {n_.shape}, 标签: {l.shape}")
