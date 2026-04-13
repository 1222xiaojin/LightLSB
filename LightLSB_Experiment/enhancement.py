import cv2
import numpy as np
import os


def enhance_decrypted_image(decrypted_img_path, output_dir):
    """
    对解密后的图像进行质量增强（基于论文 LightLSB 的方法）

    参数:
    decrypted_img_path: 解密后图像的路径 (通常是黑白/灰度图)
    output_dir: 处理结果保存的目录
    """

    # 1. 读取解密图像
    # 注意：解密图像可能是二值图(0/255)或灰度图。OpenCV 默认读取为 BGR，这里用 IMREAD_GRAYSCALE
    img = cv2.imread(decrypted_img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"无法读取图像：{decrypted_img_path}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("开始图像增强流程...")

    # =====================================================================
    # 步骤 1: 高斯滤波 (Gaussian Filtering)
    # 论文依据: 算法1 第10行
    # 目的: 减小背景噪声的影响
    # =====================================================================
    # 论文未指定具体核大小，通常使用 (5, 5) 或 (3, 3)
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "1_gaussian_blur.png"), img_gaussian)
    print("1. 高斯滤波完成")

    # =====================================================================
    # 步骤 2: 双边滤波 (Bilateral Filtering)
    # 论文依据: 算法1 第11-13行 & 公式(5)
    # 目的: 消除噪声和失真，同时保留边缘细节信息
    # 参数说明:
    #   d: 邻域直径 (对应论文算法1中的 radius)
    #   sigmaColor: 颜色空间的标准差
    #   sigmaSpace: 坐标空间的标准差 (对应论文算法1中的 sigma)
    # =====================================================================

    # 根据论文 Table 1 和 3.1节实验设置，这里提供一组经验参数
    # 论文 Table 1 显示网络结构包含卷积层，但具体滤波参数在 3.1节提及了 radius, sigma
    # 我们使用中等强度的滤波参数
    d = 9  # 邻域直径
    sigma_color = 75  # 颜色标准差
    sigma_space = 75  # 空间标准差

    img_bilateral = cv2.bilateralFilter(img_gaussian, d, sigma_color, sigma_space)
    cv2.imwrite(os.path.join(output_dir, "2_bilateral_filter.png"), img_bilateral)
    print("2. 双边滤波完成")

    # =====================================================================
    # 步骤 3: 自适应直方图均衡 (Adaptive Histogram Equalization)
    # 论文依据: 2.2节末尾 & 算法1 第14行
    # 目的: 解决图像偏黑问题，创造一致的亮度条件 (CLAHE 比普通直方图均衡更好)
    # =====================================================================

    # 创建 CLAHE 对象
    # 论文提到使用 "自适应直方图均衡"，OpenCV 中对应的是 createCLAHE
    # clipLimit 控制对比度的限制（防止过度增强噪声），tileGridSize 影响局部区域大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img_enhanced = clahe.apply(img_bilateral)

    # 保存最终恢复的图像
    final_path = os.path.join(output_dir, "FINAL_restored_image.png")
    cv2.imwrite(final_path, img_enhanced)
    print(f"3. 直方图均衡完成")
    print(f"处理完成！最终图像已保存至: {final_path}")

    return img_enhanced


# =============================================
# 【使用示例】
# 请将这里的路径替换为你实际的解密图像路径
# =============================================

if __name__ == "__main__":
    # 输入：你的解密图像路径
    decrypted_image_path = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\three\\decrypted_result.png"  # 替换为你的解密图路径

    # 输出：保存增强过程图片的文件夹
    output_folder = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\output"

    try:
        result_img = enhance_decrypted_image(decrypted_image_path, output_folder)

        # 可选：在窗口中实时查看效果（仅限本地调试）
        # cv2.imshow('Final Enhanced Image', result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"处理过程中发生错误: {e}")