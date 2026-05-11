# OUC_wangle 项目

## 项目简介

此项目包含了毕业论文《基于图像触觉传感器的边缘跟随方法及缺陷检测应用》中全部的代码、数据集、绘图

## 项目结构

```
OUC_wangle/
├── code/                # 代码目录
│   ├── segment/         # 图像分割
│   ├── resnet/          # ResNet回归
│   ├── RL/              # 强化学习
│   ├── pix2pix/         # 图像翻译
│   └── yolo/            # YOLO 目标检测
├── dataset/             # 数据集目录
│   ├── segment/         # 图像分割数据集
│   ├── resnet/          # 回归数据集
│   ├── pix2pix/         # 图像翻译数据集
│   └── yolo/            # YOLO 目标检测数据集
└── plot/                # 绘图目录
    ├── edge_following/  # 边缘跟随
    ├── yolo_exp/        # yolo实验
    └── edge_defect/     # 边缘跟随与缺陷检测结合应用
```

## 代码说明

- **segment**：将触觉图像转换为分割图像
- **resnet**：从分割图像预测出边缘角度
- **RL**：在仿真环境中执行边缘跟随任务，后续可迁移到现实环境中
- **pix2pix**：将现实中触觉图像翻译为仿真中深度图像
- **yolo**：用于视觉图像和触觉图像的缺陷检测

## 数据集说明

- **segment**：包含触觉图像、分割图像
- **resnet**：包含分割图像，命名带有边缘角度
- **pix2pix**：包含触觉图像、深度图像
- **yolo**：包含缺陷的视觉图像、触觉图像

## 绘图说明

- **edge following**：包含三种边缘跟随方法，以及不同实验物体的跟随路径txt数据
- **yolo_exp**：包含不同版本的yolo，在触觉图像和视觉图像上的训练结果csv数据
- **edge_defect**：包含结合应用实验结果，显示缺陷的实际位置和真实位置

## 安装库版本
建议单独安装，有利于排查版本冲突
```
pyhon=3.8.20
torch=1.13.1+cu116       #依赖于显卡型号
torchvision=0.14.1+cu116 #依赖于显卡型号
tqdm=4.66.1
pandas=1.1.5
numpy=1.23.1
matplotlib=3.5.1
six=1.6.10
opensimplex=0.4.5.1
opencv-python=4.10.0.84
pybullet=3.2.6
stable_baselines3=1.7.0

```



