
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from tqdm import tqdm
from model import resnet18,resnet34,resnet50
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

# 自定义加权均方误差损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # 计算每个变量的加权均方误差
        loss = self.weights * (predictions - targets) ** 2
        return loss.mean()


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, y_min=None, y_max=None):
        """
        初始化数据集
        :param image_dir: 存放图片的文件夹路径
        :param transform: 图像的预处理方法
        :param y_min: 标签的最小值，用于归一化
        :param y_max: 标签的最大值，用于归一化
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.y_min = y_min  # 最小值
        self.y_max = y_max  # 最大值

    def extract_label(self, filename):
        """
        从文件名提取标签
        :param filename: 文件名，格式为 '1_1.png' 或 '-2_1.png' 等
        :return: 标签值
        """
        label_str = filename.split('.')[0].split('_')[0]  # 提取文件名前的第一个数字部分
        label = float(label_str)
        return torch.tensor(label, dtype=torch.float32)

    def normalize(self, y):
        """
        归一化函数
        :param y: 原始标签
        :return: 归一化后的标签
        """
        if self.y_min is not None and self.y_max is not None:
            return (y - self.y_min) / (self.y_max - self.y_min)
        return y

    def denormalize(self, y_norm, device):
        """
        反归一化函数
        :param y_norm: 归一化的标签
        :param device: 设备
        :return: 原始范围的标签
        """
        if self.y_min is not None and self.y_max is not None:
            y_min = self.y_min.to(device)
            y_max = self.y_max.to(device)
            return y_norm * (y_max - y_min) + y_min
        return y_norm

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: 图像和归一化后的标签
        """
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 打开图像并提取标签
        image = Image.open(image_path).convert("RGB")
        label = self.extract_label(image_name)  # 提取标签

        # 归一化标签
        label = self.normalize(label)

        # 处理图像
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using {device} device.")

    # 定义图像的预处理
    data_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#默认
        #transforms.Normalize([0.388, 0.399, 0.374], [0.037, 0.036, 0.056])#RGB图
        transforms.Normalize([0.506, 0.506, 0.506], [0.500, 0.500, 0.500])#分割图
    ])

    # 标签的最小值和最大值（假设已知）

    # 最小值和最大值（用于归一化）
    y_min = torch.tensor([-180.0], dtype=torch.float32)
    y_max = torch.tensor([180.0], dtype=torch.float32)


    # 图像和标签路径
    #image_dir = r'/media/disk1/wl/ResNet_tactile/real_all'  # 存放图片的文件夹，图片文件名为1.jpg, 2.jpg...500.jpg
    image_dir = r'/media/disk1/wl/ResNet_tactile/unet/masks_rename'
    #image_dir = r'/media/disk1/wl/ResNet_tactile/unet/masks_rename_add'
    #image_dir = r'/media/disk1/wl/ResNet_tactile/unet/masks_rename_delete'

    # 加载数据集
    full_dataset = ImageDataset(image_dir=image_dir, transform=data_transform, y_min=y_min, y_max=y_max)

    # 划分训练集和验证集（90% 训练，10% 验证）
    train_size = int(0.90 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建 DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 加载 ResNet-18 模型
    net = resnet50()
    model_weight_path = "./resnet50-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # 修改全连接层进行回归任务
    in_channel = net.fc.in_features
    output_dim = 1  # 假设回归任务输出1个连续值
    net.fc = nn.Linear(in_channel, output_dim)
    net.to(device)

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = torch.nn.DataParallel(net,device_ids=[0, 1])

    # # 为每个变量设置权重（例如，设置第一个变量权重为2，第二个和第三个变量权重为1）
    # weights = torch.tensor([2.0, 1.0, 1.0], device=device)
    # # 使用自定义的加权损失函数
    # loss_function = WeightedMSELoss(weights)
    # 使用均方误差损失函数 (MSELoss)
    loss_function = nn.MSELoss()

    epochs = 100

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    #optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    #epochs = 100
    best_val_loss = float('inf')
    #save_path = './resNet18_regression_tactile.pth'
    save_path = './resNet50_test.pth'
    train_steps = len(train_loader)

    train_losses = []
    val_losses = []

    # 开始训练
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))  # 计算回归任务的损失
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.5f}"

        epoch_loss = running_loss / train_steps
        train_losses.append(epoch_loss)

        print(f'Epoch [{epoch+1}/{epochs}], train_loss: {running_loss/train_steps:.5f}')

        # 验证集上评估
        net.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_outputs = net(val_images.to(device))

                # 反归一化模型的输出，返回原始范围
                val_outputs_original = full_dataset.denormalize(val_outputs, val_outputs.device)
                val_labels_original = full_dataset.denormalize(val_labels, val_labels.device)

                # 收集所有输出和标签
                all_outputs.append(val_outputs_original.cpu().numpy())
                all_labels.append(val_labels_original.cpu().numpy())

                val_loss += loss_function(val_outputs, val_labels.to(device)).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{epochs}], val_loss: {avg_val_loss:.5f}')

        # 将所有输出和标签合并
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)


        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #torch.save(net.state_dict(), save_path)
            torch.save(net.module.state_dict(), save_path)#多GPU

    print('Training Finished')

    # 训练完成后的评估和绘图
    net.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = net(val_images.to(device))

            # 反归一化模型的输出，返回原始范围
            val_outputs_original = full_dataset.denormalize(val_outputs, val_outputs.device)
            val_labels_original = full_dataset.denormalize(val_labels, val_labels.device)

            # 收集所有输出和标签
            all_outputs.append(val_outputs_original.cpu().numpy())
            all_labels.append(val_labels_original.cpu().numpy())

    # 将所有输出和标签合并
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算均方误差,平均绝对误差和 R 方
    mse = mean_squared_error(all_labels, all_outputs, multioutput='raw_values')
    mae = mean_absolute_error(all_labels, all_outputs, multioutput='raw_values')
    r2 = r2_score(all_labels, all_outputs, multioutput='raw_values')


    for i in range(1):
        print(f'Variable {i+1} - MSE: {mse[i]:.4f} ,MAE: {mae[i]:.4f}, R2: {r2[i]:.4f}')

        # 绘制散点图
        plt.figure(figsize=(14, 14))
        plt.scatter(all_labels[:, i], all_outputs[:, i], alpha=0.5,s=100)
        plt.plot([all_labels[:, i].min(), all_labels[:, i].max()], [all_labels[:, i].min(), all_labels[:, i].max()],
                 'k--', lw=2)
        plt.xlabel('True Values',fontsize=30)
        plt.ylabel('Predicted Values',fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.title(f'Scatter Plot',fontsize=30)
        plt.grid(True)
        plt.savefig(f'scatter_plot.png')
        plt.close()


    epochs_range = range(epochs)
    # Training Loss
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.close()

    # Validation Loss
    plt.figure()
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Validation Loss')
    plt.savefig('validation_loss.png')
    plt.close()

if __name__ == '__main__':
    main()