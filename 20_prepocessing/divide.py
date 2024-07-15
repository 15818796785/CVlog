import os
import shutil
import tqdm

# 源图片文件夹路径
source_folder = "../../../img_align_celeba/img_align_celeba"

# 目标文件夹路径
target_folder = "../20_GeorgiaTechFaces/dataset"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 读取图像编号和人物序号的映射文件
label_mapping_file = "../identity_CelebA.txt"  # 替换为你的映射文件路径
image_labels_dict = {}

with open(label_mapping_file, 'r') as f:
    for line in f:
        img_name, label = line.strip().split()
        image_labels_dict[img_name] = label

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
    part_name = f"part_{i + 1}"
    part_path = os.path.join(target_folder, part_name)
    os.makedirs(part_path, exist_ok=True)

    # 计算当前子文件夹的图片数量
    num_images_in_current_part = images_per_folder + (1 if i < remainder else 0)

    # 获取当前子文件夹的图片文件名
    current_image_files = image_files[start_index:start_index + num_images_in_current_part]

    # 按照人物分类创建子文件夹并移动图片
    for img_file in tqdm.tqdm(current_image_files, desc=f'Moving images to {part_name}'):
        person_id = image_labels_dict[img_file]
        person_folder = os.path.join(part_path, person_id)
        os.makedirs(person_folder, exist_ok=True)

        source_path = os.path.join(source_folder, img_file)
        target_path = os.path.join(person_folder, img_file)
        shutil.move(source_path, target_path)

    # 更新起始索引
    start_index += num_images_in_current_part
