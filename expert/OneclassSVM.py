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

X,X_t,y_test = read_separate_set(processedset_path)
X_train = []
X_test = []
count = 0
for index1,item in enumerate(X):
    for index, i in enumerate(item):
        print(i.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(i, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i,hog_image =  hog(i, orientations=9, pixels_per_cell=(7, 7),cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
       # i = i.flatten()

        # 使用 exposure.rescale_intensity 来增强显示效果

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))



        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        if index == 0 and index1 == 0: plt.savefig("Hog_crop_in_OneClassSVM/or=9_pc=7_cb=1.png")
        # 等待用户按 Enter 继续
        plt.close()
      #  input("Press Enter to continue...")
        X_train.append(i)

for item in X_t:
    for i in item:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i,hog_image =  hog(i, orientations=9, pixels_per_cell=(7, 7),cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
       # i = i.flatten()
        X_test.append(i)


X_train = np.array(X_train)
X_test = np.array(X_test)

# for subject_name in tqdm(os.listdir(processedset_path), desc='reading processed images'):
#     if os.path.isdir(os.path.join(processedset_path, subject_name)):
#         subject_images_dir = os.path.join(processedset_path, subject_name)
#         temp_x_list = []
#         for img_name in os.listdir(subject_images_dir):
#             if img_name.endswith('.jpg'):
#                 img_path = os.path.join(subject_images_dir, img_name)
#                 img = cv2.imread(img_path, flags=0)
#                 # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
#                 X.append(x_feature)

# employee_features = X[:450]
# outsider_features = X[450:]
# y_test_employee = ['Accepted'] * 450
# y_test_outsider = ['Rejected'] * 300
# y_test = y_test_employee + y_test_outsider

# scaler = StandardScaler()
# X_train = scaler.fit_transform(employee_features)
#
# param_grid = {
#     'nu': [0.01, 0.05, 0.1, 0.2, 0.5],  # 通常从很小的值开始尝试
#     'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # 包括自动和手动设定的gamma
#     'kernel': ['rbf', 'linear', 'poly']  # 尝试不同的核函数
# }
#
# # 创建 OneClassSVM 对象
# oc_svm = OneClassSVM()
# grid_search = GridSearchCV(oc_svm, param_grid, cv=5, scoring='accuracy')
#
# # 训练模型
# grid_search.fit(X_train, y_test_employee)  # y_train 是用于交叉验证的标签
#
# # 训练单类SVM
# # oc_svm = OneClassSVM(kernel='rbf', gamma='scale').fit(X_train_scaled)
#
# # 在测试数据上进行预测，这里的测试数据可以包含外来者的面部图像
# X_test_scaled = scaler.transform(X)
# y_pred = oc_.predict(X)
# # 将预测结果转换为 'Accepted' 或 'Rejected'
# y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred]
# # 计算准确率

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
oc_svm = OneClassSVM(gamma='auto',nu=0.01,kernel='rbf')
oc_svm.fit(X_train)


# 在测试数据上进行预测
# X_test_scaled = scaler.transform(employee_features + outsider_features)
# X_test_pca = pca.transform(employee_features+outsider_features)
y_pred = oc_svm.predict(X_test)
y_test = np.repeat(y_test,5)
y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred]
y_test = ['Accepted' if x == 1 else 'Rejected' for x in y_test]
# 假设 y_test 是一个包含整数的列表


# 打印乘以 5 后的结果
print(y_test)

# 输出预测结果
print("y_pred",y_pred)
print("y_test",y_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")


