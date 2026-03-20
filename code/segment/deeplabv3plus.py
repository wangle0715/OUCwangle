import torch
import torch.nn as nn
import torchvision.models as models

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()

        # ASPP模块的各个分支
        self.aspp1 = ASPPConv(in_channels, out_channels, rates[0])
        self.aspp2 = ASPPConv(in_channels, out_channels, rates[1])
        self.aspp3 = ASPPConv(in_channels, out_channels, rates[2])
        self.aspp4 = ASPPConv(in_channels, out_channels, rates[3])
        self.aspp5 = ASPPPooling(in_channels, out_channels)

        # 最后的卷积层
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 各个分支的计算
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        x5 = nn.functional.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)

        # 拼接各个分支
        x = torch.cat((x1, x2, x3, x4, x5), 1)

        # 最后的卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class CustomEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class CustomDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomDecoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class DeepLabV3PlusCustom(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3PlusCustom, self).__init__()

        # 自定义的Encoder和ASPP模块
        self.encoder = CustomEncoder(in_channels, 256)
        rates = [6, 12, 18, 24]
        self.aspp = ASPP(256, 256, rates)

        # 上采样层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 使用普通的 Upsample

        # 自定义的Decoder
        self.decoder = CustomDecoder(256, 256)

        # 附加的卷积层
        self.conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder和ASPP模块
        x = self.encoder(x)
        x = self.aspp(x)

        # 上采样和Decoder
        x = self.up1(x)
        x = self.decoder(x)

        # 附加卷积层
        x = self.conv(x)

        return x

# # 创建自定义DeepLabV3+模型
# num_classes = 2  # 分割类别数（前景和背景）
# custom_deeplab_model = DeepLabV3PlusCustom(3, num_classes)  # 输入通道数为3
#
# # 设置输入数据格式
# input_data = torch.randn(1, 3, 160, 320)  # 输入数据格式为torch.Size([1, 3, 160, 320])
#
# # 模型推理
# output_mask_custom = custom_deeplab_model(input_data)
#
# # 输出的分割掩码图像
# print("Output Mask Shape (Custom):", output_mask_custom.shape)


