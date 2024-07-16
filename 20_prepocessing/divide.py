import os
import shutil
import tqdm
from collections import defaultdict

# 源图片文件夹路径
source_folder = "../20_GeorgiaTechFaces/img_align_celeba/img_align_celeba"

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

# 统计每个人的照片数量
person_image_count = defaultdict(int)
for img_file in image_files:
    person_id = image_labels_dict[img_file]
    person_image_count[person_id] += 1

# 筛选出照片数量超过10张的人物
eligible_persons = {person_id for person_id, count in person_image_count.items() if count > 10}

# 按照人物分类创建子文件夹并移动图片
for img_file in tqdm.tqdm(image_files, desc='Moving images'):
    person_id = image_labels_dict[img_file]
    if person_id in eligible_persons:
        person_folder = os.path.join(target_folder, person_id)
        os.makedirs(person_folder, exist_ok=True)

        source_path = os.path.join(source_folder, img_file)
        target_path = os.path.join(person_folder, img_file)
        shutil.move(source_path, target_path)
