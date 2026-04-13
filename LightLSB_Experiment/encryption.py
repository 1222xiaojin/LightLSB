import math
import random
import cv2
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte, view_as_blocks
import os

# ===================== 把你上面的所有代码粘贴到这里 =====================
def save_image(img_arr, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Image.fromarray(img_arr)
    img.save(output_path)

def blocks_2x2(img_en_block, share1_point, share2_point):
    block1 = np.ones((2, 2), dtype=np.uint8) * 255
    block2 = np.copy(block1)
    block_index = np.argwhere(img_en_block == 0)
    block_index_list = [(row, col) for row, col in block_index]

    if len(block_index_list) == 4 and share1_point == 0 and share2_point == 0:
        block1_index, one = random_index(block_index_list, 3)
        block2_index, other = random_index(block1_index, 2)
        block2_index = block2_index + one
        for index in block1_index:
            block1[index] = 0
        for index in block2_index:
            block2[index] = 0
    elif len(block_index_list) == 4 and share1_point == 0 and share2_point == 255:
        block1_index, one = random_index(block_index_list, 3)
        block2_index, other = random_index(block1_index, 1)
        block2_index = block2_index + one
        for index in block1_index:
            block1[index] = 0
        for index in block2_index:
            block2[index] = 0
    elif len(block_index_list) == 4 and share1_point == 255 and share2_point == 0:
        block2_index, one = random_index(block_index_list, 3)
        block1_index, other = random_index(block2_index, 1)
        block1_index = block1_index + one
        for index in block1_index:
            block1[index] = 0
        for index in block2_index:
            block2[index] = 0
    elif len(block_index_list) == 4 and share1_point == 255 and share2_point == 255:
        block1_index, block2_index = random_index(block_index_list, 2)
        for index in block1_index:
            block1[index] = 0
        for index in block2_index:
            block2[index] = 0
    elif len(block_index_list) == 3 and share1_point == 0 and share2_point == 0:
        block1 = img_en_block
        block2 = img_en_block
    elif len(block_index_list) == 3 and share1_point == 0 and share2_point == 255:
        block1 = img_en_block
        block2_index, other = random_index(block_index_list, 2)
        for index in block2_index:
            block2[index] = 0
    elif len(block_index_list) == 3 and share1_point == 255 and share2_point == 0:
        block2 = img_en_block
        block1_index, other = random_index(block_index_list, 2)
        for index in block1_index:
            block1[index] = 0
    elif len(block_index_list) == 3 and share1_point == 255 and share2_point == 255:
        block1_index, one = random_index(block_index_list, 2)
        block2_index, other = random_index(block1_index, 1)
        block2_index = block2_index + one
        for index in block1_index:
            block1[index] = 0
        for index in block2_index:
            block2[index] = 0
    else:
        print("ERROR")
    return block1, block2

def random_index(my_list, n):
    selected_elements = random.sample(my_list, n)
    remaining_elements = [element for element in my_list if element not in selected_elements]
    return selected_elements, remaining_elements

def no_pixels_expand_meaning_share(img_en, colored_share1_path, colored_share2_path):
    colored_share1 = Image.open(colored_share1_path).convert('L').convert('RGB')
    colored_share2 = Image.open(colored_share2_path).convert('L').convert('RGB')

    colored_share1 = colored_share1.resize(tuple(int(element / 2) for element in img_en.shape[:2]))
    colored_share2 = colored_share2.resize(tuple(int(element / 2) for element in img_en.shape[:2]))

    share1_ht = np.array(colored_share1.convert("1"), dtype=np.uint8) * 255
    share2_ht = np.array(colored_share2.convert("1"), dtype=np.uint8) * 255

    img_en_block = view_as_blocks(img_en, (2, 2))

    share1 = np.ones_like(img_en) * 255
    share2 = np.ones_like(img_en) * 255
    share1_block = view_as_blocks(share1, (2, 2))
    share2_block = view_as_blocks(share2, (2, 2))

    for i in range(img_en_block.shape[0]):
        for j in range(img_en_block.shape[1]):
            share1_block[i, j], share2_block[i, j] = blocks_2x2(
                img_en_block[i, j], share1_ht[i, j], share2_ht[i, j]
            )
    return share1, share2

# ===================== 下面是【使用示例】，你只需要改路径 =====================
if __name__ == '__main__':
    # 1. 读取 加密图（必须是二值黑白图）
    img_en = np.array(Image.open("C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\one\\Aaron_Patterson_0001.jpg").convert("1"), dtype=np.uint8) * 255

    # 2. 两张有意义分享图的路径
    share1_path = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\one\\Aaron_Guiel_0001.jpg"
    share2_path = "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\one\\Aaron_Eckhart_0001.jpg"

    # 3. 执行视觉密码分块生成
    share1, share2 = no_pixels_expand_meaning_share(img_en, share1_path, share2_path)

    # 4. 保存结果
    save_image(share1, "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\two\\share1_output.png")
    save_image(share2, "C:\\Users\\xiaojin\\PycharmProjects\\LightLSB_Experiment\\test\\two\\share2_output.png")

    print("生成完成！输出到 output/ 文件夹")