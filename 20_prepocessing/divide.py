import os
import shutil
import tqdm

# 源图片文件夹路径
source_folder = "../20_GeorgiaTechFaces/img_align_celeba/img_align_celeba"

# 目标文件夹路径
target_folder = "../20_GeorgiaTechFaces/dataset"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 获取所有图片文件名
image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# 计算每个子文件夹应该包含的图片数量
num_images = len(image_files)
images_per_folder = num_images // 10
remainder = num_images % 10

# 分配图片到子文件夹
start_index = 0
for i in range(10):
    # 为当前子文件夹创建路径
    subfolder_name = f"part_{i + 1}"
    subfolder_path = os.path.join(target_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 计算当前子文件夹的图片数量
    num_images_in_current_folder = images_per_folder + (1 if i < remainder else 0)

    # 获取当前子文件夹的图片文件名
    current_image_files = image_files[start_index:start_index + num_images_in_current_folder]

    # 移动图片到当前子文件夹
    for img_file in tqdm.tqdm(current_image_files, desc=f'Moving images to {subfolder_name}'):
        source_path = os.path.join(source_folder, img_file)
        target_path = os.path.join(subfolder_path, img_file)
        shutil.move(source_path, target_path)

    # 更新起始索引
    start_index += num_images_in_current_folder
