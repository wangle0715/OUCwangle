import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator,Generator128
import utils
import argparse
import os
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='tactile', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=128, help='input size')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_test_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'

# 自定义数据集类，只加载单文件夹的图像
class DatasetFromFolder(data.Dataset):
    def __init__(self, input_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.input_filenames = sorted(os.listdir(input_dir), key=lambda x: int(x.split('.')[0]))
        self.input_dir = input_dir
        self.transform = transform

    def __getitem__(self, index):
        # 加载输入图像
        input_path = os.path.join(self.input_dir, self.input_filenames[index])
        input_img = Image.open(input_path).convert('RGB')

        # 应用自定义的transform
        if self.transform:
            input_img = self.transform(input_img)

        return input_img, self.input_filenames[index]

    def __len__(self):
        return len(self.input_filenames)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize(params.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 测试数据
#test_input_folder = '/media/disk1/wl/pix2pix-master/tactile_test/real'
test_input_folder = '/media/disk1/wl/pix2pix-master/Data/tactile/trainB'
test_target_folder = '/media/disk1/wl/pix2pix-master/tactile_test/sim'

# 创建结果保存目录
if not os.path.exists(test_target_folder):
    os.makedirs(test_target_folder)

# 加载测试数据
test_data = DatasetFromFolder(input_dir=test_input_folder, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=params.batch_size, shuffle=False)

# 加载生成器模型
G = Generator128(3, params.ngf, 3)
G = G.to(device)
G.load_state_dict(torch.load(model_dir + 'generator_param100.pth'))

G = torch.nn.DataParallel(G, device_ids=[0, 1])
G.eval()

# 生成结果并保存
with torch.no_grad():
    for i, (input, filename) in enumerate(test_data_loader):
        x_ = input.to(device)
        gen_image = G(x_)
        gen_image = gen_image.cpu().data

        # 保存生成的图像
        for j in range(gen_image.size(0)):
            save_path = os.path.join(test_target_folder, filename[j])
            save_image((gen_image[j] + 1) / 2, save_path)  # 将[-1, 1]范围还原为[0, 1]

        print('%d images are generated and saved.' % (i + 1))


