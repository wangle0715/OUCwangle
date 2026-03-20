import os

# 设置图片所在文件夹的路径
folder_path = r'E:\pix2pix-master\Data\facades\testA'  # 替换为你的图片文件夹路径

# 获取文件夹中所有文件名
file_names = os.listdir(folder_path)

# 遍历文件夹中的每一个文件
for file_name in file_names:
    # 只处理以 '_A.jpg' 结尾的文件
    if file_name.endswith('_A.jpg'):
        # 获取文件名中的数字部分
        new_name = file_name.split('_')[0] + '.jpg'  # 获取数字并去除 '_A'

        # 构建完整的原始文件路径和新的文件路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {file_name} -> {new_name}')

print("Renaming completed!")