import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm
from train_test_split import train_split
from train_test_split import test_split


# 定义HOG特征提取函数
def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features, hog_image


processedset_path = "../20_GeorgiaTechFaces/Crop_1/part_1"
masked_processedset_path = "../20_GeorgiaTechFaces/Maskedcrop_1/part_2"

# 读取并处理图像数据
X_train = []
y_train = []
X_test = []
y_test = []

# # 读取不戴口罩的图像数据作为训练集
# for i, subject_name in enumerate(tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images')):
#     if os.path.isdir(os.path.join(processedset_path, subject_name)):
#         subject_images_dir = os.path.join(processedset_path, subject_name)
#         img_names = os.listdir(subject_images_dir)[:10]  # 取前10张图像
#         for img_name in img_names:
#                 img_path = os.path.join(subject_images_dir, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
#                 img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
#                 features, hog_image = extract_hog_features(img)
#                 X_train.append(features)
#                 y_train.append(i + 1)  # 使用i作为标签
#
# # 读取戴口罩的图像数据作为测试集
# for i, subject_name in enumerate(tqdm.tqdm(os.listdir(masked_processedset_path), desc='reading masked images')):
#     if os.path.isdir(os.path.join(masked_processedset_path, subject_name)):
#         subject_images_dir = os.path.join(masked_processedset_path, subject_name)
#         img_names = os.listdir(subject_images_dir)[-5:]
#         for img_name in img_names:
#                 img_path = os.path.join(subject_images_dir, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
#                 img = cv2.resize(img, (64, 64))  # 调整图像大小（可选）
#                 features, hog_image = extract_hog_features(img)
#                 X_test.append(features)
#                 y_test.append(i + 1)

X_train, y_train = train_split(processedset_path)
X_test, y_test = test_split(masked_processedset_path)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# # 标准化数据
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# 定义 k 值的范围
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
accuracies = []

# 训练并测试 kNN 模型
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 进行预测
    y_pred = knn.predict(X_test)

    # 在没有测试集的真实标签的情况下，无法直接计算准确率
    # 这里只是显示预测结果
    for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
        print(f"Image {i + 1}: True Label = {true_label}, Predicted Label = {pred_label}")

    # 计算准确率（假设有真实标签y_test）
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"k: {k}, Accuracy: {accuracy:.3f}")

#绘制 k 值与准确率的关系图（如果有真实标签y_test）
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o')
for i, (k, accuracy) in enumerate(zip(k_values, accuracies)):
    plt.text(k, accuracy, f"{accuracy:.2f}", fontsize=12, ha='center')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Neighbors (k) for Face Recognition")
plt.grid()

#保存图像并显示
save_path = "../bonus_accuracy/kNN"
os.makedirs(save_path, exist_ok=True)
plot_save_path = os.path.join(save_path, "kNN_ConvertGrayscaleprocessedset_1_k_vs_accuracy.png")
plt.savefig(plot_save_path)
plt.show()

print("length_of_X_train:{}".format(len(X_train)))
print("length_of_X_test:{}".format(len(X_test)))
