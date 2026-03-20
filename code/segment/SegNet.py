import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder_additional = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # 解码器部分
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 添加Upsample操作
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        size_1 = x.size()
        x = self.encoder(x)
        x, indices_1 = self.pool1(x)
        x = self.encoder_additional(x)
        x, indices_2 = self.pool2(x)

        x = self.unpool(x, indices_2)
        x = self.decoder(x)
        return x

# # 定义输入数据
# input_data = torch.randn(1, 3, 160, 320)
#
# # 定义SegNet模型
# num_classes = 2  # 两个类别，前景和背景
# segnet_model = SegNet(num_classes)
#
# # 前向传播得到输出
# output_mask = segnet_model(input_data)
#
# # 打印输出掩码的形状
# print(output_mask.shape)

