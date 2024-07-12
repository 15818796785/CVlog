import os
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import tqdm

processedset_path = "../GeorgiaTechFaces/Dataset_1"

X_processed = []
y = []
X_train = []
y_train = []
X_test = []
y_test = []
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
                # 将前10张图片作为训练集，后5张图片作为测试集
        if len(X_processed) >= 15:
            X_train.extend(X_processed[:10])
            y_train.extend([1] * 10)
            X_test.extend(X_processed[10:15])
            y_test.extend([1] * 5)

# 添加外部人员的图片到测试集
for outsider_img in X_processed:
    X_test.append(outsider_img)
    y_test.append(0)

# 将数据划分为员工（y=1）和外部人员（y=0）
X_employee = [X_processed[i] for i in range(len(y)) if y[i] == 1]
X_outsider = [X_processed[i] for i in range(len(y)) if y[i] == 0]

# # 划分测试集（所有）
# X_test = X_processed
# y_test = y

# 定义 k 值的范围
k_values = [1, 3, 5, 7, 9]
distances_thresholds = [1, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]
# 用于收集所有距离的列表
all_distances = []

# 创建存储准确率的二维数组
accuracies = np.zeros((len(k_values), len(distances_thresholds)))

# 训练并测试 one-class kNN 模型
cms = []
save_path = '../Accuracy/kNN'
subject_folder = os.path.join(save_path, 'Dataset_1')
os.makedirs(subject_folder, exist_ok=True)
for i, k in enumerate(k_values):
    for j, d in enumerate(distances_thresholds):
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
        knn.fit(X_employee)

        y_pred = []
        for x in X_test:
            distances, indices = knn.kneighbors([x])
            all_distances.extend(distances[0])  # 添加所有 k 个距离
            # 如果最近的邻居属于员工，则接受，否则拒绝
            if np.mean(distances) < d:
                y_pred.append(1)  # ACCEPT
            else:
                y_pred.append(0)  # REJECT

        # 将预测结果转换为接受或拒绝
        y_pred_accept_reject = ["ACCEPT" if pred == 1 else "REJECT" for pred in y_pred]
        y_test_accept_reject = ["ACCEPT" if true == 1 else "REJECT" for true in y_test]

        # 计算准确率
        accuracy = np.mean(np.array(y_pred_accept_reject) == np.array(y_test_accept_reject))
        accuracies[i, j] = accuracy
        print(f"k: {k}, distance threshold: {d}, Accuracy: {accuracy:.3f}")

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test_accept_reject, y_pred_accept_reject, labels=["ACCEPT", "REJECT"])
        cms.append(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ACCEPT", "REJECT"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (k={k}, distance threshold={d})")
        cm_save_path = os.path.join(subject_folder, f"confusion_matrix_k{k}_d{d}.png")
        plt.savefig(cm_save_path)
        plt.close()


# 计算距离值的统计信息
min_distance = np.min(all_distances)
max_distance = np.max(all_distances)
mean_distance = np.mean(all_distances)

print(f"Minimum distance: {min_distance}")
print(f"Maximum distance: {max_distance}")
print(f"Mean distance: {mean_distance}")

# 绘制 k 值与不同距离阈值下的准确率关系
plt.figure(figsize=(12, 8))
for i, k in enumerate(k_values):
    plt.plot(distances_thresholds, accuracies[i], marker='o', label=f'k={k}')
    for j, d in enumerate(distances_thresholds):
        plt.text(d, accuracies[i, j], f"{accuracies[i, j]:.2f}", fontsize=9, ha='right')
plt.xlabel("Distance Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Distance Threshold for different k values")
plt.legend()
plt.grid()

save_path = "../Accuracy/kNN"
plot_save_path = os.path.join(save_path, "kNN_Dataset_1_k_vs_accuracy.png")
plt.savefig(plot_save_path)
plt.show()
