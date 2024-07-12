import os

import cv2
from matplotlib import pyplot as plt
from skimage.exposure import exposure

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm
import numpy as np

from CVlog.expert.read_separate_set import read_separate_set

processedset_path = "../GeorgiaTechFaces/Crop_1"

X, X_t, y_test = read_separate_set(processedset_path)
X_train = []
X_test = []
count = 0
for index1, item in enumerate(X):
    for index, i in enumerate(item):
        print(i.shape)
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(i, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        i2 = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i2, hog_image = hog(i2, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=True,
                            transform_sqrt=True)
        # i = i.flatten(
        # 使用 exposure.rescale_intensity 来增强显示效果


        # 颜色直方图特征

        hist = np.concatenate([np.histogram(hsv_image[:, :, i], bins=24, range=(0, 256))[0] for i in range(3)])
        # 特征融合
        i = np.concatenate((i2, hist))
        print(i.shape)

        X_train.append(i)

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # 计算每个通道的颜色直方图
        hist_channels = [np.histogram(hsv_image[:, :, i], bins=24, range=(0, 256))[0] for i in range(3)]

        # 为每个通道创建一个图形
        channels = ['Hue', 'Saturation', 'Value']
        colors = ['r', 'g', 'b']  # 使用不同颜色以区分每个通道

        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(hist_channels):
            plt.subplot(1, 3, i + 1)  # 创建3个子图
            plt.bar(range(24), hist, color=colors[i])
            plt.title(channels[i])
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
            plt.xlim([0, 23])

        plt.tight_layout()

        if index == 0 and index1 == 0:
            plt.savefig("HSV_images_in_OneClassSVM/bins=24.png")
            cv2.imwrite("HSV_images_in_OneClassSVM/bins=24_HSV.png", hsv_image)
        # 等待用户按 Enter 继续
        plt.close("all")

    #  input("Press Enter to continue...")

for item in X_t:
    for i in item:
        i2 = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i2, hog_image = hog(i2, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=True,
                            transform_sqrt=True)
        hsv_image = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)
        # 颜色直方图特征
        hist = np.concatenate([np.histogram(hsv_image[:, :, i], bins=24, range=(0, 256))[0] for i in range(3)])
        # 特征融合
        i = np.concatenate((i2, hist))
        # i = i.flatten()
        X_test.append(i)

X_train = np.array(X_train)
X_test = np.array(X_test)

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

# 假设已经有了 X_train 和 X_test 的面部特征
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(employee_features)

# 初始化 PCA，选择主成分的数量，例如2个
# pca = PCA(n_components=6)
#
# # 对标准化后的数据进行 PCA 转换
# data_pca = pca.fit_transform(X_train)


# 创建 OneClassSVM 对象
oc_svm = OneClassSVM(gamma='auto', nu=0.01, kernel='rbf')
oc_svm.fit(X_train)

# 在测试数据上进行预测
# X_test_scaled = scaler.transform(employee_features + outsider_features)
# X_test_pca = pca.transform(employee_features+outsider_features)
y_pred = oc_svm.predict(X_test)
y_test = np.repeat(y_test, 5)
y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred]
y_test = ['Accepted' if x == 1 else 'Rejected' for x in y_test]
# 假设 y_test 是一个包含整数的列表


# 打印乘以 5 后的结果
print(y_test)

# 输出预测结果
print("y_pred", y_pred)
print("y_test", y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
