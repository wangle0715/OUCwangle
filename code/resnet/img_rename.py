import os

def rename_images(folder_path):
    x = -178  # 前缀初始值
    y_start = -4  # 每组起始y值
    y_increment = 2  # y值递增
    
    # 获取文件夹内的所有文件名，并按数字顺序排序
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    
    for i, file in enumerate(files):
        group_index = i // 5  # 当前文件属于第几组
        x_offset = group_index * 2  # 每组x值递增2
        current_x = x + x_offset
        
        y = y_start + (i % 5) * y_increment  # 当前y值
        
        new_name = f"{current_x}_{y}.png"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed {file} to {new_name}")

# 使用示例
folder_path = "/media/disk1/wl/ResNet_tactile/unet/masks_predict_rename"  # 替换为您的文件夹路径
rename_images(folder_path)
