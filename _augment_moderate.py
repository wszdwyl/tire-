# coding:utf-8
"""
对 test1 中现有中度磨损(moderate)图片做数据增强，每张生成 3 张变体。
新增图片命名延续现有编号，保存到 test1 目录。
"""
import os
import cv2
import numpy as np

os.chdir(r"D:\大赛\7.26版\Detector")
DATA_DIR = './Data/test/test1/'

# 读取现有中度磨损图
existing = sorted([f for f in os.listdir(DATA_DIR)
                   if f.lower().startswith('moderate') and f.lower().endswith('.jpg')])
print(f"现有中度磨损图: {len(existing)} 张")

# 确定起始编号（取现有最大编号+1）
import re
nums = []
for f in existing:
    m = re.search(r'moderate\s*\((\d+)\)', f, re.IGNORECASE)
    if m:
        nums.append(int(m.group(1)))
start_num = max(nums) + 1 if nums else 1
print(f"新增图片从 moderate ({start_num}).jpg 开始编号")


def aug_flip_bright(img):
    """水平翻转 + 亮度+10%"""
    img = cv2.flip(img, 1)
    return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)


def aug_rotate_contrast(img, angle=10):
    """旋转+10° + 对比度+10%"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(img.astype(np.float32) * 1.1 + 10, 0, 255).astype(np.uint8)


def aug_rotate_dark_blur(img, angle=-10):
    """旋转-10° + 亮度-10% + 轻微高斯模糊"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.5)


aug_funcs = [aug_flip_bright, aug_rotate_contrast, aug_rotate_dark_blur]

count = 0
cur = start_num
for fname in existing:
    path = os.path.join(DATA_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"  跳过无法读取: {fname}")
        continue
    for func in aug_funcs:
        aug_img = func(img)
        out_name = f"moderate ({cur}).jpg"
        cv2.imwrite(os.path.join(DATA_DIR, out_name), aug_img)
        cur += 1
        count += 1

print(f"\n完成: 新增 {count} 张增强图")
print(f"中度磨损总数: {len(existing) + count} 张")
