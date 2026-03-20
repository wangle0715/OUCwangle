import os
import torch
from torchvision import transforms
from PIL import Image
from unet import UNet
# from deeplabv3plus import DeepLabV3PlusCustom
# from SegNet import SegNet
from utils.data_loading import BasicDataset
from evaluate import dice_loss
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def predict_img(net, full_img, device, scale_factor=0.5):
    net.eval()
    #输入图像预处理，将图像转换为张量
    # print(f'full_img channels: {full_img.shape}')
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    #print(f'img channels: {img.shape}')
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #print(f'img channels: {img.shape}')

    #传递给神经网络进行预测
    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)[0]
        #用于图像的转换
        tf = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()])
        full_mask = tf(probs.cpu()).squeeze()
    return (full_mask > 0.5).numpy()
def mask_to_image(mask: np.ndarray):#将一个NumPy数组转换为PIL图像
    #检查掩码的维度是否为2，如果是则将其转换为灰度图像；否则将掩码转换为一个RGB图像。
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
def plot_three(img, mask, mask_gt):#画图，包含输入图像，预测语义分割图，真值语义分割图
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Predicted mask')
    ax[1].imshow(mask)
    ax[2].set_title('Ground truth mask')
    ax[2].imshow(mask_gt)
    plt.show()

def test(net, device, test_data_dir, test_data_masks, output_dir, scale_factor=0.5):
    net.to(device=device)
    net.eval()
    test_files = os.listdir(test_data_dir)
    for filename in test_files:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_data_dir, filename)
            img = Image.open(img_path)

            mask = predict_img(net=net, full_img=img, scale_factor=scale_factor, device=device)
            # print(mask.shape)  # (2, 320, 640)
            mask = mask[0, :, :]
            mask = 1 - mask  # 反转掩码，将背景和前景交换
            # print(mask.shape)  # (320, 640)
            # 转换
            result = mask_to_image(mask)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            result.save(os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_OUT.png'))
            # 加载真值语义分割图
            mask_gt_path = os.path.join(test_data_masks, f'{os.path.splitext(filename)[0]}.png')
            mask_gt = np.array(imageio.imread(mask_gt_path))
            # print(mask_gt.shape) # (1280, 1918, 3)
            mask = mask.astype(np.float32)
            mask_gt = mask_gt.astype(np.float32)
            # print(f'mask shape: {mask.shape}')
            # print(f'mask_gt shape: {mask_gt.shape}')

            # 计算dice score

            diceloss= 1+dice_loss(torch.from_numpy(mask), torch.from_numpy(mask_gt))
            dice_score =1-diceloss

            print(dice_score)

            plot_three(img, mask, mask_gt)




if __name__ == '__main__':
    # 设置路径
    model_path = './checkpoints/checkpoint_epoch_unet.pth'
    # model_path = './checkpoints_segnet/checkpoint_epoch10.pth'
    #model_path = './checkpoints_deeplabv3plus/checkpoint_epoch10.pth'
    test_data_imgs = './test_img/imgs/'
    test_data_masks = './test_img/masks/'
    output_dir = './test_results/'

  
    # 实例化模型
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    # net = SegNet(num_classes=2)
    #net = DeepLabV3PlusCustom(in_channels=3, num_classes=2)
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    # 执行测试
    test(net, device, test_data_imgs, test_data_masks, output_dir, scale_factor=0.5)