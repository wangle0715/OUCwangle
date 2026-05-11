import matplotlib.pyplot as plt
import os
import numpy as np

# ================================
# 读取位移数据
# ================================

# file_path = os.path.join('E:/module2/txt/defect', 'xy_circle_concave.txt')
file_path = os.path.join('E:/OUC_wangle/plot/edge_following/txt/rl_yxw', 'xy_circle.txt')
#file_path = os.path.join('E:/module2', 'xy_circle.txt')
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
# 生成圆形参考轨迹
# ================================

r = 50
center_x, center_y = 50, 0
theta = np.linspace(0, 2 * np.pi, 300)
circle_x = center_x + r * np.cos(theta)
circle_y = center_y + r * np.sin(theta)
circle_points = list(zip(circle_x, circle_y))

# ================================
# 计算偏差（欧几里得距离）
# ================================

distances = []
for px, py in points[:-1]:  # 忽略最后回到原点的点
    min_dist = min(np.hypot(px - cx, py - cy) for cx, cy in circle_points)
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
ax.plot(circle_x, circle_y, linestyle='-', color='blue', label='Circle',linewidth=2)

# 起点终点标记
ax.plot(x_coords[0], y_coords[0], 'g^', markersize=15, markeredgecolor='black', label='Start')
ax.plot(x_coords[-2], y_coords[-2], 'rs', markersize=15, markeredgecolor='black', label='End')

# 添加一个朝下的箭头（从起点 (0, 0) 朝向 (0, -20)）
ax.annotate(
    '',                             # 没有文字
    xy=(-5, -30),                    # 箭头指向的终点
    xytext=(-5, 0),                  # 箭头起点
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
ax.set_xticks(np.arange(-40, 150, 20))
ax.set_yticks(np.arange(-90, 100, 20))

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


