import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# ---------- 读取位移 ----------
file_path = os.path.join('E:/module2/txt/rl_yxw', 'xy_fish.txt')
displacements = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                displacements.append(tuple(map(float, parts)))
            except ValueError:
                print(f"无法转换: {line}")
        else:
            print(f"格式错误，跳过: {line}")

# ---------- 生成轨迹 ----------
points = [(0, 0)]
cx, cy = 0, 0
for dx, dy in displacements:
    cx += dx
    cy += dy
    points.append((cx, cy))
points.append((0, 0))
x_coords, y_coords = zip(*points)

# ---------- 绘图 ----------
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ax.set_facecolor('#eaeaf2')

# 蓝色轨迹线
ax.plot(x_coords, y_coords, color='purple', linewidth=2, label='Path', zorder=5)

# 方向箭头
# 蓝色原始点
ax.scatter(x_coords, y_coords, color='purple', s=40, zorder=5)

ax.annotate('', xy=(-30, 5), xytext=(0, 5),
            arrowprops=dict(arrowstyle='-|>', color='blue', lw=2,
                            shrinkA=0, shrinkB=0, mutation_scale=20))

# ---------- 起点/终点（最上层） ----------
ax.scatter(x_coords[0],  y_coords[0],  marker='^', s=15**2, color='green',
           edgecolors='black', zorder=10, label='Start')
ax.scatter(x_coords[-2], y_coords[-2], marker='s', s=15**2, color='red',
           edgecolors='black', zorder=10, label='End')

# 坐标轴与网格
ax.set_xlabel('X(mm)', fontsize=35)
ax.set_ylabel('Y(mm)', fontsize=35)
ax.grid(True)
ax.set_aspect('equal', 'box')
ax.set_xticks(np.arange(-60, 70, 20))
ax.set_yticks(np.arange(-140, 50, 20))
ax.tick_params(axis='both', labelsize=35)
ax.legend(loc='upper right', fontsize=25, frameon=True)

# ---------- 0. 把透明 PNG 插到最底层 ----------
img_path = r'E:\module2\img\img_result\soft\fish_bg.jpg'  # ← 你的透明背景图
arr = mpimg.imread(img_path)                          # 0-1 浮点 RGBA

# 控制图在轴系里的“左下角”位置、缩放系数
x0, y0, zoom = 3, -58, 0.155      # ← 自己调，单位 mm
imagebox = OffsetImage(arr, zoom=zoom, zorder=0)   # zorder=0 保证最底
ab = AnnotationBbox(imagebox, (x0, y0),
                    xycoords='data', frameon=False)
ax.add_artist(ab)
# -------------------------------------------------

plt.tight_layout()
plt.show()