import os
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from read_separate_set import read_separate_set
import tqdm

processedset_path = "../GeorgiaTechFaces/Dataset_1"

X_processed = []
y = []
X_train = []
y_train = []
X_test = []
y_test = []
# 读取并处理图像数据

X_train, X_test, y_test = read_separate_set(processedset_path)
# Flatten X_train properly
X_train_flat = np.array([img.flatten() for sublist in X_train for img in sublist])
X_test_flat = np.array([img.flatten() for sublist in X_test for img in sublist])
y_test = [item for item in y_test for _ in range(5)]
k = 1
# 定义距离阈值的范围
distances_thresholds = [1, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]

# 存储所有距离的列表
all_distances = []

# 创建存储准确率的二维数组
accuracies = np.zeros((len(distances_thresholds),))

# 训练并测试 one-class kNN 模型
cms = []
save_path = '../Accuracy/kNN/NO_feature'
subject_folder = os.path.join(save_path, 'Dataset_1')
os.makedirs(subject_folder, exist_ok=True)

for j, d in enumerate(distances_thresholds):
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
    knn.fit(X_train_flat)
    y_pred = []
    for x in X_test_flat:
        distances, indices = knn.kneighbors([x])
        all_distances.extend(distances[0])  # 添加所有 k 个距离
        # 如果最近的邻居属于员工，则接受，否则拒绝
        if distances[0][0] < d:
            y_pred.append(1)  # ACCEPT
        else:
            y_pred.append(0)  # REJECT

    # 将预测结果转换为接受或拒绝
    y_pred_accept_reject = ["ACCEPT" if pred == 1 else "REJECT" for pred in y_pred]
    y_test_accept_reject = ["ACCEPT" if true == 1 else "REJECT" for true in y_test]

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[j] = accuracy
    print(f"Distance threshold: {d}, Accuracy: {accuracy:.3f}")

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

# 绘制距离阈值与准确率的关系图
plt.figure(figsize=(12, 8))
plt.plot(distances_thresholds, accuracies, marker='o')
for j, d in enumerate(distances_thresholds):
    plt.text(d, accuracies[j], f"{accuracies[j]:.2f}", fontsize=9, ha='right')
plt.xlabel("Distance Threshold")
plt.ylabel("Accuracy")
plt.title(f"Accuracy vs Distance Threshold (k={k})")
plt.grid()

plot_save_path = os.path.join(save_path, "Dataset_1_kNN_accuracy_vs_distance_threshold.png")
plt.savefig(plot_save_path)
plt.show()