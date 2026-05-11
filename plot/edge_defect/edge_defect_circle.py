import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches



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
# 绘图
# ================================

fig, ax = plt.subplots()

# 设置背景颜色
fig.patch.set_facecolor('white')         # 整体背景色
ax.set_facecolor('#eaeaf2')              # 坐标区域背景色


# 绘制圆形轨迹
ax.plot(circle_x, circle_y, linestyle='-', color='blue', label='Path',linewidth=2)

#凹陷-真值concave

# ax.plot(50, -47.5, 'o', markersize=16, markerfacecolor='red',markeredgecolor='black', label='circle_concave')
# ax.plot(50, 47.5, 's', markersize=16, markerfacecolor='y', markeredgecolor='black', label='square_concave')
# # 添加一个长方形作为终点标记（例如长宽为 6x3）
# line = patches.Rectangle((95 - 5, 0 - 1.25), 10, 2.5, linewidth=1,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
#                          edgecolor='black', facecolor='darkorange', label='line_concave')
# ax.add_patch(line)

#凹陷-预测concave

# x1,y1=51.23, -46.42
# x2,y2=48.65, 46.74
# x3,y3=95.96,1.12
# w1,h1=5.35,5.33
# w2,h2=5.32,5.35
# w3,h3=10.21,2.76
# circle = patches.Rectangle((x1 - w1/2, y1 - h1/2), w1, h1, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
#                          edgecolor='red', facecolor='none', label='circle_concave')
# # 添加一个长方形作为终点标记（例如长宽为 6x3）
# square = patches.Rectangle((x2 - w2/2, y2 - h2/2),w2 , h2, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
#                          edgecolor='y', facecolor='none', label='square_concave')
# # 添加一个长方形作为终点标记（例如长宽为 6x3）
# line = patches.Rectangle((x3 - w3/2, y3 - h3/2), w3, h3, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
#                          edgecolor='darkorange', facecolor='none', label='line_concave')
# ax.add_patch(circle)
# ax.add_patch(square)
# ax.add_patch(line)

# 凸起-真值convex
#
# ax.plot(50, -47.5, 'o', markersize=16, markerfacecolor='lightpink',markeredgecolor='black', label='circle_convex')
# ax.plot(50, 47.5, 's', markersize=16, markerfacecolor='lime', markeredgecolor='black', label='square_convex')
# # 添加一个长方形作为终点标记（例如长宽为 6x3）
# line = patches.Rectangle((95 - 5, 0 - 1.25), 10, 2.5, linewidth=1,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
#                          edgecolor='black', facecolor='gold', label='line_convex')
# ax.add_patch(line)

#凸起-预测convex

x1,y1=48.98, -48.55
x2,y2=49.01, 48.72
x3,y3=93.87,-1.42
w1,h1=5.24,5.22
w2,h2=5.19,5.18
w3,h3=10.19,2.81

circle = patches.Rectangle((x1 - w1/2, y1 - h1/2), w1, h1, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
                         edgecolor='lightpink', facecolor='none', label='circle_convex')
# 添加一个长方形作为终点标记（例如长宽为 6x3）
square = patches.Rectangle((x2 - w2/2, y2 - h2/2),w2 , h2, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
                         edgecolor='lime', facecolor='none', label='square_convex')
# 添加一个长方形作为终点标记（例如长宽为 6x3）
line = patches.Rectangle((x3 - w3/2, y3 - h3/2), w3, h3, linewidth=3,#(3 - 3, 1 - 1.5) 是左下角坐标，保证以 (3, 1) 为中心;  6, 3 是宽和高；
                         edgecolor='gold', facecolor='none', label='line_convex')
ax.add_patch(circle)
ax.add_patch(square)
ax.add_patch(line)

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




# 添加图例ax.legend()
ax.legend(loc='upper right',fontsize=20, frameon=True)

# 自适应排版
plt.tight_layout()

plt.show()


