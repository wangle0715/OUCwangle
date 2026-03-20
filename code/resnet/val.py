import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import resnet18
import numpy as np
import sys
import cv2



def predict(image, model_weight_path, y_min, y_max):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 定义图像的预处理
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.506, 0.506, 0.506], [0.500, 0.500, 0.500])
    ])

    # 将NumPy数组转换为PIL图像并处理

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)

    image = data_transform(image).unsqueeze(0)  # 增加一个维度以匹配batch size为1

    # 加载 ResNet-18 模型
    net = resnet18()
    in_channel = net.fc.in_features
    output_dim = 1  # 假设回归任务输出1个连续值
    net.fc = nn.Linear(in_channel, output_dim)
    net.to(device)

    # 加载预训练权重
    assert os.path.exists(model_weight_path), f"File {model_weight_path} does not exist."
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()

    # 进行预测
    with torch.no_grad():
        output = net(image.to(device))

        y_min = y_min.to(device)
        y_max = y_max.to(device)

        # 反归一化模型的输出，返回原始范围
        outputs_original = (output * (y_max - y_min) + y_min).item()

    return outputs_original  # 返回预测值
 

if __name__ == '__main__':
    # 单张图像的路径
    path=r"/media/disk1/wl/ResNet_tactile/crop_180/11.png"
    #path=r"/media/disk1/wl/ResNet_tactile/unet/masks_predict/8.png"
    #path=r"/media/disk1/wl/ResNet_tactile/unet/masks_rename/-100_4.png"
    #path=r"/media/disk1/wl/ResNet_tactile/crop1_unet/6.png"

    #path=r"/media/disk1/wl/ResNet_tactile/3.png"

    image = Image.open(path).convert("RGB")

    y_min = torch.tensor([-180.0], dtype=torch.float32)
    y_max = torch.tensor([180.0], dtype=torch.float32)

    # 模型权重路径
    model_weight_path = r"/media/disk1/wl/ResNet_tactile/resNet18_regression_tactile_add.pth"  # 替换为你的模型权重路径

    # 进行预测并保存
    prediction = predict(image, model_weight_path, y_min, y_max)
    print(f"Prediction: {prediction:.1f}")