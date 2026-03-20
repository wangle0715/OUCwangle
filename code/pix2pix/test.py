import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator
import utils
import argparse
import os
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import random



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
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

class DatasetFromTwoFolders(data.Dataset):
    def __init__(self, input_dir, target_dir, transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromTwoFolders, self).__init__()
        self.input_filenames = sorted(os.listdir(input_dir),key=lambda x: int(x.split('.')[0]))
        self.target_filenames = sorted(os.listdir(target_dir), key=lambda x: int(x.split('.')[0]))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

        # 检查两个文件夹的图像数量是否匹配
        assert len(self.input_filenames) == len(self.target_filenames), \
            "Input and target folder must have the same number of images!"

    def __getitem__(self, index):
        # 加载输入图像和目标图像
        input_path = os.path.join(self.input_dir, self.input_filenames[index])
        target_path = os.path.join(self.target_dir, self.target_filenames[index])

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # 图像预处理
        if self.resize_scale:
            input_img = input_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target_img = target_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size)
            y = random.randint(0, self.resize_scale - self.crop_size)
            input_img = input_img.crop((x, y, x + self.crop_size, y + self.crop_size))
            target_img = target_img.crop((x, y, x + self.crop_size, y + self.crop_size))

        if self.fliplr and random.random() < 0.5:
            input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
            target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 应用自定义的transform
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

    def __len__(self):
        return len(self.input_filenames)


# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])

# Test data
# test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=test_transform)
# test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
#                                                batch_size=params.batch_size,
#                                                shuffle=False)
test_input_folder = '/media/disk1/wl/pix2pix-master/Data/facades/testB'
test_target_folder = '/media/disk1/wl/pix2pix-master/Data/facades/testA'

test_data = DatasetFromTwoFolders(input_dir=test_input_folder,target_dir=test_target_folder, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
                            
#test_input, test_target = test_data_loader.__iter__().__next__()


# Load model
G = Generator(3, params.ngf, 3)
G = G.to(device)
G.load_state_dict(torch.load(model_dir + 'generator_param200.pth'))

G = torch.nn.DataParallel(G,device_ids=[0, 1])

G.eval() 

#Test 
with torch.no_grad():
    for i, (input, target) in enumerate(test_data_loader):
    # input & target image data
        x_ = input.to(device)
        y_ = target.to(device)
        gen_image = G(x_)
        gen_image = gen_image.cpu().data

    # Show result for test data
        utils.plot_test_result(input, target, gen_image, i, training=False, save=True, save_dir=save_dir)

        print('%d images are generated.' % (i + 1))

