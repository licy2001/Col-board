# -*- coding: utf-8 -*-
# @Author  : Licy
# @Time    : 2024/1/23 0:45
# @Email   : licy0089@gmail.com
# @Software: PyCharm

import os
import random
from PIL import Image, ImageFilter, ImageEnhance


# 将图像分割成patches
def split_image_into_patches(img, patch_size):
    patches = []
    for i in range(0, img.width, patch_size[0]):
        for j in range(0, img.height, patch_size[1]):
            patch = img.crop((i, j, i + patch_size[0], j + patch_size[1]))
            patches.append(patch)
    return patches


# # 在图像上mask掉一些patches
# def mask_patches_on_image(img, patches_to_mask, patch_size=(32, 32)):
#     masked_img = img.copy()
#     for patch in patches_to_mask:
#         i, j = patch
#         for x in range(i * patch_size[0], (i + 1) * patch_size[0]):
#             for y in range(j * patch_size[1], (j + 1) * patch_size[1]):
#                 masked_img.putpixel((x, y), 0)  # 将像素设置为黑色
#     return masked_img

# from PIL import Image, ImageDraw


# def mask_patches_on_image(img, patches_to_mask, patch_size):
#     masked_img = img.copy()
#     transparent = (0, 0, 0, 0)  # 完全透明的颜色
#     draw = ImageDraw.Draw(masked_img)
#     for patch in patches_to_mask:
#         i, j = patch
#         x1, y1 = i * patch_size[0], j * patch_size[1]
#         x2, y2 = (i + 1) * patch_size[0], (j + 1) * patch_size[1]
#         draw.rectangle([x1, y1, x2, y2], fill=transparent)  # 将指定区域填充为透明色
#     del draw
#     return masked_img


def mask_patches_on_image(img, patches_to_mask, patch_size, color):
    masked_img = img.copy()

    for patch in patches_to_mask:
        i, j = patch
        for x in range(i * patch_size[0], (i + 1) * patch_size[0]):
            for y in range(j * patch_size[1], (j + 1) * patch_size[1]):
                masked_img.putpixel((x, y), color)  # 将像素设置为白色
    return masked_img


# 在图像上对一些patches进行退化处理
def degrade_patches_on_image(img, patches_to_degrade, patch_size):
    degraded_img = img.copy()
    for patch in patches_to_degrade:
        i, j = patch
        patch_img = degraded_img.crop(
            (
                i * patch_size[0],
                j * patch_size[1],
                (i + 1) * patch_size[0],
                (j + 1) * patch_size[1],
            )
        )
        # 对patch进行均值滤波
        patch_img = patch_img.filter(ImageFilter.BLUR())
        degraded_img.paste(patch_img, (i * patch_size[0], j * patch_size[1]))
    return degraded_img


def get_mask_img(img, patch_size):
    # 计算图像的行数和列数
    rows, cols = img.size

    # 计算每个patch的大小
    # patch_rows = rows // 16  # 640 / 16 = 40
    # patch_cols = cols // 12  # 480 / 12 = 40

    patch_rows = rows // 32  # 128 / 8 = 16
    patch_cols = cols // 32  # 128 / 8 = 16

    # 创建patch索引
    # patch_indices = [(i // 12, i % 12) for i in range(16 * 12)]
    patch_indices = [(i // 4, i % 4) for i in range(4 * 4)]

    # 随机选择160个patches
    selected_patches = random.sample(patch_indices, 14)

    # 创建两个mask图像
    # masked_img1 = mask_patches_on_image(
    #     img, selected_patches[:100], (patch_rows, patch_cols, ), color=black
    # )
    # masked_img2 = mask_patches_on_image(
    #     img, selected_patches[100:], (patch_rows, patch_cols), color=white
    # )
    masked_img1 = mask_patches_on_image(img, selected_patches[:7], patch_size, color=black)
    masked_img2 = mask_patches_on_image(img, selected_patches[7:], patch_size, color=black)
    return masked_img1, masked_img2


if __name__ == "__main__":
    # patch_size = (16, 12)
    patch_size = (32, 32)
    white = (255, 255, 255)  # 白色的RGB值
    black = (0, 0, 0)  # 白色的RGB值

    # 输入和输出文件夹的路径
    input_folder_path = "/data2/wait/bisheCode/DDPM_Fusion/dataset/TXCJ128/val/ImageS"
    output_pathA = "/data2/wait/bisheCode/DDPM_Fusion/dataset/TXCJ/val/ImageA"
    output_pathB = "/data2/wait/bisheCode/DDPM_Fusion/dataset/TXCJ/val/ImageB"
    # 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_pathA, exist_ok=True)
    os.makedirs(output_pathB, exist_ok=True)

    # 获取输入文件夹中所有图像文件的路径
    image_files = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # 循环处理每张图像
    for image_file in image_files:
        # 构建输入图像的完整路径
        input_image_path = os.path.join(input_folder_path, image_file)

        # 构建输出图像的完整路径，保持相同的文件名
        output_image_pathA = os.path.join(output_pathA, image_file)
        output_image_pathB = os.path.join(output_pathB, image_file)
        # 打开图像
        img = Image.open(input_image_path)

        Image_A, Image_B = get_mask_img(img, patch_size)
        # 保存处理后的图像到输出文件夹
        Image_A.save(output_image_pathA)
        Image_B.save(output_image_pathB)

        print(f"Processed")
