import matplotlib.pyplot as plt
import os
import numpy as np

# ================================
# 读取位移数据
# ================================


#file_path = os.path.join('E:/module2', 'xy_clover.txt')
file_path = os.path.join('E:/module2/txt/resnet', 'xy_clover.txt')
displacements = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                x, y = map(float, parts)
                displacements.append((x, y))
            except ValueError:
                print(f"无法将以下内容转换为浮动数值: {line}")
        else:
            print(f"格式错误的行，跳过: {line}")

# ================================
# 生成轨迹坐标
# ================================

points = [(0, 0)]
current_x, current_y = 0, 0

for dx, dy in displacements:
    current_x += dx
    current_y += dy
    points.append((current_x, current_y))

points.append((0, 0))  # 回到原点

x_coords, y_coords = zip(*points)

# ================================
# 四叶草参考轨迹
# ================================

# ================================
# 生成四叶草参考轨迹
# ================================
r = 25
x1, y1 = 0, -25
x2, y2 = -25, -50
x3, y3 = 0, -75
x4, y4 = 25, -50

# 生成四个半圆，角度分别是 0, pi/2, pi, 3pi/2
theta1 = np.linspace(0, np.pi, 300)        # 第一个半圆（从右侧）
theta2 = np.linspace(np.pi/2, 3*np.pi/2, 300)  # 第二个半圆（从上侧）
theta3 = np.linspace(np.pi, 2*np.pi, 300)      # 第三个半圆（从左侧）
theta4 = np.linspace(3*np.pi/2, 5*np.pi/2, 300)  # 第四个半圆（从下侧）

# 四个半圆的 x 和 y 坐标
circle_x1 = x1 + r * np.cos(theta1)
circle_y1 = y1 + r * np.sin(theta1)

circle_x2 = x2 + r * np.cos(theta2)
circle_y2 = y2 + r * np.sin(theta2)

circle_x3 = x3 + r * np.cos(theta3)
circle_y3 = y3 + r * np.sin(theta3)

circle_x4 = x4 + r * np.cos(theta4)
circle_y4 = y4 + r * np.sin(theta4)

# # 合并四个半圆的坐标
circle_x = np.concatenate([circle_x1, circle_x2, circle_x3, circle_x4])
circle_y = np.concatenate([circle_y1, circle_y2, circle_y3, circle_y4])



circle_points = list(zip(circle_x, circle_y))

# ================================
# 计算偏差（欧几里得距离）
# ================================

distances = []

for px, py in points[:-1]:  # 忽略最后回到原点的点
    min_dist = min(np.hypot(px - cx, py - cy) for cx, cy in circle_points)  # 计算到四叶草轨迹最近点的距离
    distances.append(min_dist)

average_distance = np.mean(distances)
max_distance = np.max(distances)

print(f"所有轨迹点到参考轨迹的平均最小距离为: {average_distance:.4f}")
print(f"所有轨迹点到参考轨迹的最大最小距离为: {max_distance:.4f}")


# 计算起点和终点之间的距离
start_x, start_y = points[0]
end_x, end_y = points[-2]  # 倒数第二个点是终点
end_to_start_dist = np.hypot(end_x - start_x, end_y - start_y)
print(f"起点和终点之间的距离为: {end_to_start_dist:.4f}")

# ================================
# 绘图
# ================================

fig, ax = plt.subplots()

# 设置背景颜色
fig.patch.set_facecolor('white')         # 整体背景色
ax.set_facecolor('#eaeaf2')              # 坐标区域背景色

# 绘制灰色轨迹线
ax.plot(x_coords, y_coords, linestyle='-', color='gray', alpha=0.5,label='Path',linewidth=2)

# 设置颜色映射范围
vmin = 0
vmax = max(distances)

# 绘制偏差热力图（颜色越红偏差越大）
sc = ax.scatter(
    x_coords[:-1], y_coords[:-1],
    c=distances,
    cmap='hot',     # 或者使用 cmap='Reds'
    s=40,           # 这里控制轨迹点的大小
    vmin=vmin,
    vmax=vmax
    #label='Deviation'
)

# 绘制圆形轨迹
ax.plot(circle_x, circle_y, linestyle='-', color='blue', label='Clover',linewidth=2)

# 起点终点标记
ax.plot(x_coords[0], y_coords[0], 'g^', markersize=15, markeredgecolor='black', label='Start')
ax.plot(x_coords[-2], y_coords[-2], 'rs', markersize=15, markeredgecolor='black', label='End')

# 添加一个朝下的箭头（从起点 (0, 0) 朝向 (0, -20)）
ax.annotate(
    '',                             # 没有文字
    xy=(-30, 5),  # 箭头指向的终点
    xytext=(0, 5),  # 箭头起点
    arrowprops=dict(
        arrowstyle='-|>',  # 更粗的箭头
        color='blue',
        lw=2,              # 线宽
        shrinkA=0, shrinkB=0,#用来控制箭头和它指向的点之间的“缩进距离”。
        mutation_scale=20  # 箭头大小
    )
)

# 图表设置
#ax.set_title('Movement Path with Deviation Heatmap')
ax.set_xlabel('X(mm)', fontsize=35)
ax.set_ylabel('Y(mm)', fontsize=35)
ax.grid(True)
ax.set_aspect('equal', 'box')

# 设置坐标刻度
ax.set_xticks(np.arange(-90, 100, 20))
ax.set_yticks(np.arange(-140, 60, 20))

# 设置坐标刻度字体大小
ax.tick_params(axis='x', labelsize=35)
ax.tick_params(axis='y', labelsize=35)


# 添加颜色条
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Error(mm)', fontsize=35)  # 设置颜色条标签的字体大小
cbar.ax.tick_params(labelsize=35)  # 设置颜色条刻度字体大小

# 添加图例ax.legend()
ax.legend(loc='upper right',fontsize=25, frameon=True)

# 自适应排版
plt.tight_layout()

plt.show()


