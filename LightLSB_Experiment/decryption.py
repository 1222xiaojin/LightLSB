import cv2
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte, view_as_blocks
import os


def block_no_expand(PIL_img):
    """
    块处理函数
    :param PIL_img: PIL图像
    :return: 处理后的图像
    """
    image = np.array(PIL_img)
    # 缩放图像强度
    image = np.interp(image, (0, 255), (0, 255 / 4))
    # 转换为二值图像
    image_ht = Image.fromarray(image).convert("1")
    # 转换为8位无符号整数格式
    image_ht = img_as_ubyte(image_ht)
    return image_ht


def post_process(decrypted_img):
    """
    后处理函数：双边滤波去噪
    :param decrypted_img: 解密后的图像
    :return: 处理后的图像
    """
    # 双边滤波去噪，对应论文2.2节
    return cv2.bilateralFilter(decrypted_img, d=9, sigmaColor=75, sigmaSpace=75)


def decrypt(share1_path, share2_path, output_path):
    """
    解密函数
    :param share1_path: 分享1路径
    :param share2_path: 分享2路径
    :param output_path: 输出路径
    :return: 解密后的图像
    """
    # 读取分享图像
    share1 = np.array(Image.open(share1_path).convert('L'))
    share2 = np.array(Image.open(share2_path).convert('L'))

    # 解密：按位与操作
    decrypted = share1 & share2

    # 后处理
    decrypted = post_process(decrypted)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, decrypted)

    return decrypted


# ====================== 你只需要修改这里的路径 ======================
if __name__ == "__main__":
    # 输入：两张分享图像路径
    SHARE1 = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\two\\share1_output.png"
    SHARE2 = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\two\\share2_output.png"

    # 输出：解密结果保存路径
    OUTPUT = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\three\\decrypted_result.png"

    # 执行解密
    result = decrypt(SHARE1, SHARE2, OUTPUT)
    print("解密完成！结果已保存至：", OUTPUT)