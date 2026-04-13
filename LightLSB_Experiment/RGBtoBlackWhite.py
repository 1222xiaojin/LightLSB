import cv2
import numpy as np
import os
import glob


def floyd_steinberg_dithering(image):
    """
    对灰度图像进行 Floyd-Steinberg 误差扩散半色调处理。
    文档中提到的误差扩散系数 (Floyd-Steinberg coefficients) 为 7/16, 3/16, 5/16, 1/16。

    Args:
        image: 输入的单通道灰度图像 (值范围 0-255)

    Returns:
        二值图像 (0 或 255)
    """
    # 转换为 float32 进行计算
    img = image.astype(np.float32)
    h, w = img.shape

    # 遍历每个像素
    for y in range(h):
        for x in range(w):
            # 1. 二值化 (文档公式 1)
            # 文档公式: h(x,y) = 0 if p(x,y)<0.5 else 1
            # 这里先计算误差，再确定输出
            old_pixel = img[y, x]
            # 将 0-255 映射到 0-1 进行阈值判断
            if old_pixel < 128:
                new_pixel = 0
            else:
                new_pixel = 255
            img[y, x] = new_pixel

            # 2. 计算量化误差 (文档公式 2)
            # e(x,y) = p(x,y) - h(x,y)
            # 这里的 p(x,y) 是标准化后的输入，但我们直接在 0-255 空间操作
            quant_error = old_pixel - new_pixel

            # 3. 扩散误差 (文档公式 3)
            # Floyd-Steinberg 系数: 7/16, 3/16, 5/16, 1/16
            if x + 1 < w:
                img[y, x + 1] += quant_error * 7 / 16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += quant_error * 3 / 16
                img[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += quant_error * 1 / 16

    return img.astype(np.uint8)


def preprocess_image(img_path):
    """
    实现文档中描述的预处理流程 (算法 1 的前半部分)。

    流程:
    1. 读取图像 (BGR -> RGB)
    2. RGB -> HSV (文档 2.1 节)
    3. 亮度重映射 (V: 0-255 -> 0-64, 文档公式 4, c=4)
    4. HSV -> RGB
    5. 转为灰度图 (为了半色调做准备)
    6. 误差扩散半色调 (文档 1.2 节, Floyd-Steinberg)

    Args:
        img_path: 输入图像路径

    Returns:
        processed_img: 处理后的二值图像 (0 或 255)
    """
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    # BGR 转 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. RGB 转 HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 3. 亮度重映射 (V 通道)
    # 文档描述: "将亮度值从 0~255 重新映射到 0~64"
    # 公式 I' = I_ln(...) / c, 其中 c=4
    img_hsv_float = img_hsv.astype(np.float32)
    img_hsv_float[:, :, 2] = (img_hsv_float[:, :, 2] / 4)  # 除以 4，将范围压缩到 0-63.75

    # 4. 转回 uint8 并转换回 RGB
    # 注意: 这里需要截断并转换类型，OpenCV 需要 uint8
    img_hsv_clipped = np.clip(img_hsv_float, 0, 255).astype(np.uint8)
    img_rgb_back = cv2.cvtColor(img_hsv_clipped, cv2.COLOR_HSV2RGB)

    # 5. 转为灰度图 (半色调通常在灰度图上进行)
    # 如果已经是二值图逻辑，这一步是必须的
    gray = cv2.cvtColor(img_rgb_back, cv2.COLOR_RGB2GRAY)

    # 6. 误差扩散半色调处理
    # 文档 1.2 节: 误差扩散 (Error Diffusion)
    binary_img = floyd_steinberg_dithering(gray)

    return binary_img


def main(input_folder, output_folder):
    """
    批量处理文件夹中的图像。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    print(f"开始处理文件夹: {input_folder}")

    for ext in extensions:
        # 获取所有匹配的文件
        search_path = os.path.join(input_folder, ext)
        files = glob.glob(search_path)

        for file_path in files:
            try:
                # 处理图像
                result_img = preprocess_image(file_path)

                # 构建输出路径
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_folder, filename)

                # 保存图像
                # OpenCV 保存图像为 BGR，但二值图无影响
                cv2.imwrite(output_path, result_img)
                print(f"已处理: {filename}")

            except Exception as e:
                print(f"处理失败 {file_path}: {str(e)}")

    print(f"处理完成，结果保存在: {output_folder}")


if __name__ == "__main__":
    # --- 请在此处修改你的输入输出路径 ---
    input_dir = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\input\\train"  # 存放原始图片的文件夹
    output_dir = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\input\\real_train"  # 存放处理后图片的文件夹
    # ---------------------------------

    main(input_dir, output_dir)