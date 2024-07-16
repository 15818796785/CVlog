import os
from tqdm import tqdm

# 源文件夹路径和目标文件夹路径
source_folder = "../20_GeorgiaTechFaces/washed_data"
compare_folder = "../20_GeorgiaTechFaces/masked"

# 获取所有人的文件夹
source_person_folders = sorted(os.listdir(source_folder))
compare_person_folders = sorted(os.listdir(compare_folder))

# 遍历每个人的文件夹进行对比
for person_folder in tqdm(source_person_folders, desc="Processing persons"):
    source_person_folder = os.path.join(source_folder, person_folder)
    compare_person_folder = os.path.join(compare_folder, person_folder)

    # 如果比较文件夹中没有这个人文件夹，跳过
    if not os.path.exists(compare_person_folder):
        continue

    # 获取两个文件夹中该人的图片文件名
    source_images = set(os.listdir(source_person_folder))
    compare_images = set(os.listdir(compare_person_folder))

    # 找出需要删除的图片
    images_to_delete = compare_images - source_images

    # 删除需要删除的图片
    for img_name in images_to_delete:
        img_path = os.path.join(compare_person_folder, img_name)
        os.remove(img_path)
        print(f"Deleted {img_path}")

print("Processing complete.")
