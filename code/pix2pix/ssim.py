import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError(f"无法读取 {image1_path} 或 {image2_path}")

    return ssim(img1, img2)

def process_images(folder1, folder2):
    ssim_values = []
    image_pairs = []

    for i in range(1, 901):  # 图片编号从 1 到 900
        img1_path = os.path.join(folder1, f"{i}.png")
        img2_path = os.path.join(folder2, f"{i}.png")

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"警告: {img1_path} 或 {img2_path} 不存在，跳过")
            continue

        similarity = calculate_ssim(img1_path, img2_path)
        ssim_values.append(similarity)
        image_pairs.append((f"{i}.png", similarity))

    if not ssim_values:
        print("未计算到任何 SSIM 值")
        return

    min_ssim = min(ssim_values)
    max_ssim = max(ssim_values)
    avg_ssim = np.mean(ssim_values)

    min_image = [name for name, value in image_pairs if value == min_ssim][0]
    max_image = [name for name, value in image_pairs if value == max_ssim][0]

    print(f"最小 SSIM: {min_ssim:.4f}, 对应图片: {min_image}")
    print(f"最大 SSIM: {max_ssim:.4f}, 对应图片: {max_image}")
    print(f"平均 SSIM: {avg_ssim:.4f}")


# 替换成你的图片文件夹路径
folder1 ='/media/disk1/wl/pix2pix-master/tactile_test/sim'
folder2 = '/media/disk1/wl/pix2pix-master/Data/tactile/trainA'

process_images(folder1, folder2)
