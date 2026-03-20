import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import resnet18


def predict(image, model_weight_path, y_min, y_max):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using {device} device.")

    # 定义图像的预处理
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.506, 0.506, 0.506], [0.500, 0.500, 0.500])
    ])

    # 处理图像
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
    # 文件夹路径
    folder_path = r"/media/disk1/wl/ResNet_tactile/crop_180" # 替换为实际的文件夹路径


    #folder_path = r"/media/disk1/wl/ResNet_tactile/crop1_unet"
    #folder_path = r"/media/disk1/wl/ResNet_tactile/unet/masks"#_predict"
    # 定义目标范围
    y_min = torch.tensor([-180.0], dtype=torch.float32)
    y_max = torch.tensor([180.0], dtype=torch.float32)

    # 模型权重路径
    #model_weight_path = r"/media/disk1/wl/ResNet_tactile/resNet18_regression_tactile1.pth"
    #model_weight_path = r"/media/disk1/wl/ResNet_tactile/resNet18_regression_tactile_predict.pth"  # 替换为实际的权重文件路径
    model_weight_path = r"/media/disk1/wl/ResNet_tactile/resNet18_regression_tactile_add.pth" 

    # 遍历文件夹中的所有图片，按照文件名排序
    all_files = os.listdir(folder_path)
    image_files = sorted([f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))], key=lambda x: int(x.split('.')[0]))

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        #print(f"Processing {image_file}...")

        # 打开图像并进行预测
        image = Image.open(image_path).convert("RGB")
        prediction = predict(image, model_weight_path, y_min, y_max)

        print(f"Prediction for {image_file}: {prediction:.1f}")
