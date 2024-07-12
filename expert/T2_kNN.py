import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import tqdm

processedset_path = "../GeorgiaTechFaces/Crop_1"

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

# 将数据划分为员工（y=1）和外部人员（y=0）
X_employee = [X_processed[i] for i in range(len(y)) if y[i] == 1]
X_outsider = [X_processed[i] for i in range(len(y)) if y[i] == 0]

# 划分测试集（所有）
X_test = X_processed
y_test = y

# 定义 k 值的范围
k_values = [1, 3, 5, 7, 9]
metric_values = ['minkowski']#, 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'hamming']
accuracies = []

# 训练并测试 one-class kNN 模型
for k in k_values:
    for m in metric_values:
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=30, metric=m, p=2)
        knn.fit(X_employee)

        y_pred = []
        for x in X_test:
            distances, indices = knn.kneighbors([x])
            # 如果最近的邻居属于员工，则接受，否则拒绝
            if np.mean(distances) < np.mean(knn.kneighbors(X_employee)[0]):
                y_pred.append(1)  # ACCEPT
            else:
                y_pred.append(0)  # REJECT

        # 将预测结果转换为接受或拒绝
        y_pred_accept_reject = ["ACCEPT" if pred == 1 else "REJECT" for pred in y_pred]
        y_test_accept_reject = ["ACCEPT" if true == 1 else "REJECT" for true in y_test]

        # 计算准确率
        accuracy = np.mean(np.array(y_pred_accept_reject) == np.array(y_test_accept_reject))
        accuracies.append(accuracy)
        print(f"k: {k}, matric: {m}, Accuracy: {accuracy:.3f}")

# 绘制 k 值与准确率的关系
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Neighbors (k) for Face Recognition")
plt.grid()

save_path = "../Accuracy/kNN"
plot_save_path = os.path.join(save_path, "kNN_Crop_1_k_vs_accuracy.png")
plt.savefig(plot_save_path)
plt.show()
