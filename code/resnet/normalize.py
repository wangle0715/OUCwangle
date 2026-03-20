import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 定义路径
#data_dir = "/media/disk1/wl/ResNet_tactile/real_all"
data_dir = "/media/disk1/wl/ResNet_tactile/unet/masks_rename"

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor()  # 转换为 Tensor，通道顺序变为 (C, H, W)
])

# 加载数据集
dataset = ImageFolderDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 初始化统计量
mean = np.zeros(3)
std = np.zeros(3)
num_pixels = 0

# 遍历数据集
for images in dataloader:
    batch_pixels = images.size(0) * images.size(2) * images.size(3)  # 批次中像素总数 (batch_size * H * W)
    num_pixels += batch_pixels
    mean += images.sum(axis=(0, 2, 3)).numpy()  # 累加每个通道的总和
    std += (images ** 2).sum(axis=(0, 2, 3)).numpy()  # 累加每个通道平方的总和

# 计算均值和标准差
mean /= num_pixels
std = np.sqrt(std / num_pixels - mean ** 2)

print(f"Mean: {mean}")
print(f"Std: {std}")
