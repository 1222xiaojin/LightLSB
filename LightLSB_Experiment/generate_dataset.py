import os
import cv2
import numpy as np
import argparse
import shutil
from itertools import combinations
import random


def generate_base_matrices(n, k):
    """生成EVCS的基矩阵S0和S1"""
    all_vectors = []
    for indices in combinations(range(n), k):
        vector = [0] * n
        for idx in indices:
            vector[idx] = 1
        all_vectors.append(vector)

    m = len(all_vectors)
    selected_indices = random.sample(range(m), m)

    S0 = np.array([all_vectors[i] for i in selected_indices], dtype=np.uint8)
    S1 = 1 - S0

    return S0, S1


def encrypt_image_evcs(image, n=2, k=2):
    """基于EVCS的图像加密方法"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    shares = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]

    S0, S1 = generate_base_matrices(n, k)

    for i in range(h):
        for j in range(w):
            pixel_val = gray[i, j]

            if pixel_val < 128:
                base_matrix = S1
            else:
                base_matrix = S0

            col_index = np.random.randint(0, base_matrix.shape[1])

            for share_idx in range(n):
                shares[share_idx][i, j] = base_matrix[share_idx, col_index] * 255

    return shares


def decrypt_image_evcs(shares, k=2):
    """EVCS解密：通过视觉叠加恢复黑白图像"""
    shares = shares[:k]

    h, w = shares[0].shape
    v_decrypted = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if any(share[i, j] == 255 for share in shares):
                v_decrypted[i, j] = 255
            else:
                v_decrypted[i, j] = 0

    return v_decrypted


def generate_blurry_images(clear_dir, blurry_dir, n=2, k=2):
    """生成基于EVCS的模糊图像并保存到指定目录"""
    os.makedirs(blurry_dir, exist_ok=True)
    processed_count = 0
    skipped_files = []

    for filename in os.listdir(clear_dir):
        # 仅处理图像文件
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        clear_path = os.path.join(clear_dir, filename)
        clear_img = cv2.imread(clear_path)

        if clear_img is None:
            skipped_files.append(filename)
            continue

        # 应用EVCS解密模拟模糊效果
        shares = encrypt_image_evcs(clear_img, n, k)
        blurry_img = decrypt_image_evcs(shares, k)

        # 保存黑白图像
        blurry_path = os.path.join(blurry_dir, filename)
        cv2.imwrite(blurry_path, blurry_img)
        processed_count += 1

    print(f"成功处理 {processed_count} 张图像")
    if skipped_files:
        print(
            f"跳过 {len(skipped_files)} 个非图像文件或无效文件: {', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}")


def split_dataset(clear_dir, blurry_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """划分数据集为训练集和验证集"""
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, 'clear', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'clear', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blurry', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blurry', 'val'), exist_ok=True)

    # 获取所有图像文件
    clear_files = [f for f in os.listdir(clear_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 计算划分点
    train_count = int(len(clear_files) * train_ratio)

    # 复制文件到相应目录
    for i, filename in enumerate(clear_files):
        src_clear = os.path.join(clear_dir, filename)
        src_blurry = os.path.join(blurry_dir, filename)

        if i < train_count:
            # 训练集
            dst_clear = os.path.join(output_dir, 'clear', 'train', filename)
            dst_blurry = os.path.join(output_dir, 'blurry', 'train', filename)
        else:
            # 验证集
            dst_clear = os.path.join(output_dir, 'clear', 'val', filename)
            dst_blurry = os.path.join(output_dir, 'blurry', 'val', filename)

        shutil.copy(src_clear, dst_clear)
        shutil.copy(src_blurry, dst_blurry)

    print(f"数据集划分完成: 训练集 {train_count} 张, 验证集 {len(clear_files) - train_count} 张")


def main():
    parser = argparse.ArgumentParser(description='生成基于EVCS的模糊图像数据集')
    parser.add_argument('--input', required=True, help='原始清晰图像目录')
    parser.add_argument('--output', required=True, help='输出数据集目录')
    parser.add_argument('--n', type=int, default=2, help='份额数量')
    parser.add_argument('--k', type=int, default=2, help='阈值k')
    args = parser.parse_args()

    # 1. 生成模糊图像
    print("1/2 正在生成基于EVCS的模糊图像...")
    blurry_temp_dir = os.path.join(args.output, 'blurry_temp')
    generate_blurry_images(args.input, blurry_temp_dir, args.n, args.k)

    # 2. 划分数据集
    print("2/2 正在划分数据集...")
    split_dataset(args.input, blurry_temp_dir, args.output)

    # 清理临时目录
    shutil.rmtree(blurry_temp_dir)
    print("✅ 数据集生成完成")


if __name__ == "__main__":
    main()