import os
import random
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import tqdm

processedset_path = "../GeorgiaTechFaces/ConvertGrayscaleMaskprocessedset_1"

X_processed = []
y = []
X_train = []
y_train = []
X_test = []
y_test = []
# 读取并处理图像数据

# LBP参数
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist
def shuffle_array(X, y):
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    X_shuffled, y_shuffled = zip(*zipped)
    return list(X_shuffled), list(y_shuffled)


def read_separate_set(path):
    X_read_train = []
    X_read_test = []
    # first 10 img in folder is for training, the rest is for testing
    for subject_name in tqdm.tqdm(os.listdir(path), desc='reading processed images'):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg') and 1 <= int(img_name.split('.')[0]) <= 10:
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                    img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
                    features = extract_lbp_features(img)
                    # cv2.resize(img, (150, 150))
                    temp_x_list.append(features)
            X_read_train.append(temp_x_list)

    for subject_name in tqdm.tqdm(os.listdir(path), desc='reading test processed images'):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg') and int(img_name.split('.')[0]) >= 11 and int(
                        img_name.split('.')[0]) <= 15:
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
                    img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
                    features = extract_lbp_features(img)
                    # cv2.resize(img, (150, 150))
                    temp_x_list.append(features)
            X_read_test.append(temp_x_list)

    # shuffle reading sets
    X_read_train, X_read_test = shuffle_array(X_read_train, X_read_test)
    X_employee = X_read_train[0:30]
    X_outsider = X_read_train[30:]

    y_employee = [1] * len(X_employee)
    y_outsider = [0] * len(X_outsider)
    y = y_employee + y_outsider
    X_test, y_test = shuffle_array(X_read_test, y)
    return X_employee, X_test, y_test



X_train, X_test, y_test = read_separate_set(processedset_path)
# Flatten X_train properly
X_train_flat = np.array([img.flatten() for sublist in X_train for img in sublist])
X_test_flat = np.array([img.flatten() for sublist in X_test for img in sublist])
y_test = [item for item in y_test for _ in range(5)]
k = 1
# 定义距离阈值的范围
distances_thresholds = [0, 0.001, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03]

# 存储所有距离的列表
all_distances = []

# 创建存储准确率的二维数组
accuracies = np.zeros((len(distances_thresholds),))

# 训练并测试 one-class kNN 模型
cms = []
save_path = '../Accuracy/kNN/LBP'
subject_folder = os.path.join(save_path, 'ConvertGrayscaleMaskprocessedset_1')
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

plot_save_path = os.path.join(save_path, "ConvertGrayscaleMaskprocessedset_1_kNN_accuracy_vs_distance_threshold.png")
plt.savefig(plot_save_path)
plt.show()