import os
import cv2
import numpy as np

def calculate_psnr(image1_path, image2_path):
    """计算两张图片的 PSNR (dB)"""
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError(f"无法读取 {image1_path} 或 {image2_path}")

    # 如果尺寸不一致，直接返回 0（或可根据需要 resize）
    if img1.shape != img2.shape:
        return 0.0

    return cv2.PSNR(img1, img2)      # 返回值单位 dB

def process_images(folder1, folder2):
    psnr_values = []
    image_pairs = []

    for i in range(1, 901):  # 图片编号 1~900
        img1_path = os.path.join(folder1, f"{i}.png")
        img2_path = os.path.join(folder2, f"{i}.png")

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"警告: {img1_path} 或 {img2_path} 不存在，跳过")
            continue

        psnr = calculate_psnr(img1_path, img2_path)
        psnr_values.append(psnr)
        image_pairs.append((f"{i}.png", psnr))

    if not psnr_values:
        print("未计算到任何 PSNR 值")
        return

    min_psnr = min(psnr_values)
    max_psnr = max(psnr_values)
    avg_psnr = np.mean(psnr_values)

    min_image = [name for name, val in image_pairs if val == min_psnr][0]
    max_image = [name for name, val in image_pairs if val == max_psnr][0]

    print(f"最小 PSNR: {min_psnr:.2f} dB, 对应图片: {min_image}")
    print(f"最大 PSNR: {max_psnr:.2f} dB, 对应图片: {max_image}")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")


# 替换成你的图片文件夹路径
folder1 = '/media/disk1/wl/pix2pix-master/tactile_test/sim'
folder2 = '/media/disk1/wl/pix2pix-master/Data/tactile/trainA'

process_images(folder1, folder2)