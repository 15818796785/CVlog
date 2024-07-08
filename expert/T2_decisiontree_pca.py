import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tqdm

processedset_path = "../GeorgiaTechFaces/Maskprocessedset_1"

X_processed = []
y = []

# 读取并处理图像数据
for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
                img_flattened = img.flatten()  # 将图像展平
                X_processed.append(img_flattened)
                y.append(1 if len(y) < 30 * len(os.listdir(subject_images_dir)) else 0)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 应用PCA进行降维
pca = PCA(n_components=50)  # 保留50个主成分，可以根据实际情况调整
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 定义 max_depth 值的范围
max_depth_values = [1, 2, 4, 8, 16]
accuracies = []

# 训练并测试决策树模型
for max_depth in max_depth_values:
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(X_train_pca, y_train)
    accuracy = dtree.score(X_test_pca, y_test)
    accuracies.append(accuracy)
    print(f"Max Depth: {max_depth}, Accuracy: {accuracy:.3f}")

# 绘制 max_depth 值与准确率的关系
plt.figure(figsize=(8, 6))
plt.plot(max_depth_values, accuracies, marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Max Depth for Face Recognition with PCA")
plt.grid()
plt.show()
