import torch
from torchvision import transforms
from model import Generator128
from PIL import Image
import argparse
import os
import cv2
import numpy as np
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
# parser.add_argument('--input_image', required=True, help='path to the input image')  # 指定单张输入图片路径
# parser.add_argument('--output_image', required=True, help='path to save the output image')  # 指定输出图片路径
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in the first layer')
parser.add_argument('--input_size', type=int, default=128, help='input size for the model')
#parser.add_argument('--model_path', required=True, help='path to the pre-trained generator model')
params = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(params.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载生成器模型
G = Generator128(3, params.ngf, 3)
G = G.to(device)
model_path = "/media/disk1/wl/pix2pix-master/tactile_model/generator_param100.pth"
G.load_state_dict(torch.load(model_path))
G.eval()

# 加载单张输入图片
input_image_path = '/media/disk1/wl/pix2pix-master/tactile_test/real/1.png'
# 使用 OpenCV 加载图片
input_image = cv2.imread(input_image_path,cv2.IMREAD_COLOR)  # BGR 格式
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
# 转换为 PIL.Image 格式
input_image_pil = Image.fromarray(input_image)
# 应用 transforms 并转换为张量
input_tensor = transform(input_image_pil).unsqueeze(0).to(device)  # 添加 batch 维度

# 生成结果并保存
with torch.no_grad():
    gen_image = G(input_tensor)
    gen_image = gen_image.cpu().data[0]  # 去掉 batch 维度

# 将张量转换为 NumPy 格式
gen_image = (gen_image + 1) / 2  # 将张量范围从 [-1, 1] 转换为 [0, 1]
gen_image = gen_image.mean(dim=0).numpy()  # 取平均值生成单通道灰度图
gen_image = (gen_image * 255).astype(np.uint8)  # 将像素值缩放到 [0, 255]
#print(gen_image.shape)
output_image_path='/media/disk1/wl/pix2pix-master/1.png'
# 保存为灰度图
cv2.imwrite(output_image_path, gen_image)


print(f"Image has been processed and saved to {output_image_path}.")

