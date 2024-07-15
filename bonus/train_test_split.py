import os
import cv2
import numpy as np
from skimage.feature import hog


# 读取并处理图像文件
def process_images_in_folder(folder_path, limit, offset=0):
    print(3)
    images = []
    labels = []

    for i, img_name in enumerate(os.listdir(folder_path)):
        if offset <= i < limit:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = np.asarray(img, dtype=np.float32)
                hog_features = hog(img, orientations=6, pixels_per_cell=(9, 9), cells_per_block=(2, 2), visualize=False,
                                   transform_sqrt=True)
                images.append(hog_features)
                labels.append(os.path.basename(folder_path))
        elif i >= limit:
            break
    return images, labels


# 训练集划分
def train_split(path):
    print(1)
    X_train = []
    y_train = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for folder in os.listdir(subdir_path):
                folder_path = os.path.join(subdir_path, folder)
                if os.path.isdir(folder_path):
                    num_images = len(os.listdir(folder_path))
                    train_limit = int(num_images / 3)

                    X_tr, y_tr = process_images_in_folder(folder_path, train_limit)
                    X_train.extend(X_tr)
                    y_train.extend(y_tr)

    return X_train, y_train


# 测试集划分
def test_split(path):
    print(2)
    X_test = []
    y_test = []

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            for folder in os.listdir(subdir_path):
                folder_path = os.path.join(subdir_path, folder)
                if os.path.isdir(folder_path):
                    num_images = len(os.listdir(folder_path))
                    test_offset = num_images - int(num_images / 3)

                    X_te, y_te = process_images_in_folder(folder_path, num_images, test_offset)
                    X_test.extend(X_te)
                    y_test.extend(y_te)

    return X_test, y_test


# # 示例用法:
# path = "20_classified_image_washed"
# X_train, y_train = train_split(path)
# X_test, y_test = test_split(path)
