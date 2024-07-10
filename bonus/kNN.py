import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import tqdm

processedset_path = "../GeorgiaTechFaces/Crop_1"
masked_processedset_path = "../GeorgiaTechFaces/Maskedcrop_1"

# 读取并处理图像数据
X_processed = []
y = []

# 读取不戴口罩的图像数据
for i, subject_name in enumerate(tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images')):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
                img_flattened = img.flatten()  # 将图像展平
                X_processed.append(img_flattened)
                y.append(i - 1)

# 读取戴口罩的图像数据作为测试集
X_processed_masked = []
for subject_name in tqdm.tqdm(os.listdir(masked_processedset_path), desc='reading masked images'):
    if os.path.isdir(os.path.join(masked_processedset_path, subject_name)):
        subject_images_dir = os.path.join(masked_processedset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
                img_flattened = img.flatten()  # 将图像展平
                X_processed_masked.append(img_flattened)

# 定义 k 值的范围
k_values = [1, 3, 5, 7, 9]
accuracies = []

# 训练并测试 kNN 模型
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_processed, y)

    # 进行预测
    y_pred = knn.predict(X_processed_masked)

    for i, pred_label in enumerate(y_pred):
        print(f"Predicted label for {i + 1}: {pred_label}")

        # 如果有测试集的真实标签，计算准确率
        # 在没有测试集的真实标签的情况下，无法直接计算准确率，可以使用其他指标进行评估
        # 这里仅作示例，假设有真实标签 y_true
    accuracy = accuracy_score(y, y_pred)
    accuracies.append(accuracy)
    print(f"k: {k}, Accuracy: {accuracy:.3f}")

# 绘制 k 值与准确率的关系图
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Neighbors (k) for Face Recognition")
plt.grid()

# 保存图像并显示
save_path = "../bonus_accuracy/kNN"
plot_save_path = os.path.join(save_path, "kNN_Crop_1_k_vs_accuracy.png")
plt.savefig(plot_save_path)
plt.show()
