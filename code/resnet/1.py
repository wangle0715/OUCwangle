
from model import resnet18
import torch
from torchviz import make_dot

# 初始化ResNet18模型
model = resnet18(num_classes=1000, include_top=True)
model.eval()

# 构造输入张量（模拟224×224的3通道图像）
x = torch.randn(1, 3, 224, 224)
y = model(x)

# 生成结构图并保存为PDF（矢量图，放大不模糊）
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("resnet18_structure", format="pdf")