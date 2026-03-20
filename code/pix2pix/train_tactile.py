import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator,Generator128, Discriminator128
import utils
import argparse
import os
#from logger import Logger
import random
from PIL import Image
import torch.utils.data as data


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='tactile', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=128, help='input size')
parser.add_argument('--resize_scale', type=int, default=144, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=128, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=100, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


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

train_input_folder = '/media/disk1/wl/pix2pix-master/Data/tactile/trainB'
train_target_folder = '/media/disk1/wl/pix2pix-master/Data/tactile/trainA'

# Train data
train_data = DatasetFromTwoFolders(
    input_dir=train_input_folder,
    target_dir=train_target_folder,
    transform=transform,
    resize_scale=params.resize_scale,
    crop_size=params.crop_size,
    fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True)
# train_data = DatasetFromFolder(data_dir, subfolder='train', direction=params.direction, transform=transform,
#                                resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
# train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
#                                                 batch_size=params.batch_size,
#                                                 shuffle=True)

test_input_folder = '/media/disk1/wl/pix2pix-master/Data/tactile/testB'
test_target_folder = '/media/disk1/wl/pix2pix-master/Data/tactile/testA'


# Test data
test_data = DatasetFromTwoFolders(input_dir=test_input_folder,
                                  target_dir=test_target_folder,
                                  transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()

# Models
G = Generator128(3, params.ngf, 3)#适用于128大小图像
D = Discriminator128(6, params.ndf, 1)

G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

G = G.to(device)
D = D.to(device)

G = torch.nn.DataParallel(G,device_ids=[0, 1])
D = torch.nn.DataParallel(D,device_ids=[0, 1])




# Set the logger#使用Logger类记录生成器和判别器的训练损失，并将其写入TensorBoard以便后续分析和可视化
# D_log_dir = save_dir + 'D_logs'
# G_log_dir = save_dir + 'G_logs'
# if not os.path.exists(D_log_dir):
#     os.mkdir(D_log_dir)
# D_logger = Logger(D_log_dir)
#
# if not os.path.exists(G_log_dir):
#     os.mkdir(G_log_dir)
# G_logger = Logger(G_log_dir)

# Loss function
BCE_loss = torch.nn.BCELoss().to(device)
L1_loss = torch.nn.L1Loss().to(device)

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0
for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []

    G.train()
    D.train()

    # training
    for i, (input, target) in enumerate(train_data_loader):

        # input & target image data
        x_ = input.to(device)
        y_ = target.to(device)

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = torch.ones_like(D_real_decision, device=device)
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = torch.zeros_like(D_fake_decision, device=device)
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))

        # ============ TensorBoard logging ============#
        # D_logger.scalar_summary('losses', D_loss.item(), step + 1)
        # G_logger.scalar_summary('losses', G_loss.item(), step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    G.eval() 

    gen_image = G(test_input.to(device)).cpu().data
    utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)

# Plot average losses
utils.plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# # Save trained parameters of model
# torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
# torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
#多GPU
torch.save(G.module.state_dict(), model_dir + 'generator_para100_test.pth')
torch.save(D.module.state_dict(), model_dir + 'discriminator_param100_test.pth')


