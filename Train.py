import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import main_model
import DataLoader

# 修复1：添加显存自动增长配置（防止OOM）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 修复2：增强数据加载的异常处理
try:
    anc_data, pos_data, neg_data, labels = DataLoader.dataloder()
except Exception as e:
    print(f"数据加载失败: {str(e)}")
    exit(1)


# 修复3：改进的Triplet Loss函数
def triplet_loss(y_true, y_pred):
    MARGIN = 1.0
    anchor = y_pred[:, :128]
    positive = y_pred[:, 128:256]
    negative = y_pred[:, 256:]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.mean(K.maximum(pos_dist - neg_dist + MARGIN, 0.0))
    return loss


# 修复4：模型构建与训练流程
try:
    # 创建模型
    model = main_model.TripletModel()

    # 编译模型（使用learning_rate替代已弃用的lr参数）
    model.compile(
        loss=triplet_loss,
        optimizer=Adam(learning_rate=1e-4)  # 关键修复点
    )

    # 训练模型（添加验证集监控）
    print("[INFO] 输入数据形状:")
    print(f"锚点数据: {anc_data.shape}")
    print(f"正样本: {pos_data.shape}")
    print(f"负样本: {neg_data.shape}")

    history = model.fit(
        [anc_data, pos_data, neg_data],
        labels,
        batch_size=8,
        epochs=7,
        validation_split=0.2,  # 添加验证集
        verbose=1
    )

    # 保存模型（添加异常处理）
    try:
        model.save("model_new.h5")
        print("[INFO] 模型保存成功")
    except Exception as e:
        print(f"模型保存失败: {str(e)}")

except Exception as e:
    print(f"模型训练失败: {str(e)}")