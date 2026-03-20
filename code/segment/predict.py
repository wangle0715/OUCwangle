import torch
import logging
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from unet import UNet
from SegNet import SegNet
from deeplabv3plus import DeepLabV3PlusCustom
# 或 SegNet / DeepLabV3PlusCustom

# ===== 设置路径 =====
MODEL_PATH = Path('/media/disk1/wl/Unet_tactile/checkpoints/checkpoint_epoch10.pth')
#MODEL_PATH = Path('/media/disk1/wl/Unet_tactile/checkpoints/checkpoint_epoch_unet.pth')
#MODEL_PATH = Path('/media/disk1/wl/Unet_tactile/checkpoints/checkpoint_epoch_segnet.pth')
#MODEL_PATH = Path('/media/disk1/wl/Unet_tactile/checkpoints/checkpoint_epoch_deeplabv3plus.pth')

IMG_DIR = Path('./crop1/')
OUTPUT_DIR = Path('./test_results/')
NUM_CLASSES = 2
USE_BILINEAR = False


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(image)


def predict():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 加载模型
    #net = UNet(n_channels=3, n_classes=2, bilinear=USE_BILINEAR)
    net = SegNet(num_classes=2)
    #net = DeepLabV3PlusCustom(in_channels=3, num_classes=2)


    net.to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    logging.info(f'Model loaded from {MODEL_PATH}')

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取图像
    img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])

    for img_path in tqdm(img_paths, desc='Predicting'):
        input_tensor = load_image(img_path).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

        with torch.no_grad():
            output = net(input_tensor)  # [1, C, H, W]
            probs = torch.sigmoid(output)
            pred_mask = (probs > 0.5).float()

        # 保存第1通道的 mask（通常是前景/目标）
        mask_np = pred_mask[0, 1].cpu().numpy()
        save_path = OUTPUT_DIR / f"{img_path.stem}.png"
        plt.imsave(save_path, mask_np, cmap='gray')

    logging.info(f"Prediction done. Saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    predict()
