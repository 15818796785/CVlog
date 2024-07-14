import os
import shutil

# 定义路径
source_dir = 'img_align_celeba'  # 替换为存储图片的源文件夹路径
dest_dir = '20_classified_image'    # 替换为输出分类图片的文件夹路径

# 确保目标文件夹存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 读取训练集文件，假设文件名为 'train.txt'
with open('identity_CelebA.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行数据
for line in lines:
    img_name, label = line.strip().split()
    label_dir = os.path.join(dest_dir, label)

    # 确保标签文件夹存在
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 源文件路径和目标文件路径
    src_file = os.path.join(source_dir, img_name)
    dest_file = os.path.join(label_dir, img_name)

    # 复制文件到目标文件夹
    shutil.copy(src_file, dest_file)

print("图片分类完成！")
