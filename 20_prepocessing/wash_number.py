import os
import shutil


def remove_folders_with_fewer_than_ten_images(parent_folder):
    # 获取所有子文件夹的名称
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        # 获取该子文件夹中的所有文件
        files = os.listdir(subfolder_path)
        # 统计.jpg图片的数量
        image_files = [f for f in files if f.endswith('.jpg')]

        if len(image_files) < 10:
            # 删除少于十张照片的文件夹
            shutil.rmtree(subfolder_path)
            print(f"Deleted folder: {subfolder_path} with {len(image_files)} images")


# 传入文件夹路径
parent_folder = "../20_GeorgiaTechFaces/dataset"
remove_folders_with_fewer_than_ten_images(parent_folder)
