import os

import cv2

from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

processedset_path = "../GeorgiaTechFaces/Processedset_1"

X = []
y = []
X_train = []

for subject_name in tqdm(os.listdir(processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, flags=0)
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                                cells_per_block=(1, 1), visualize=False)
                X.append(x_feature)

employee_features = X[:450]
outsider_features = X[450:]
y_test_employee = ['Accepted'] * 450
y_test_outsider = ['Rejected'] * 300
y_test = y_test_employee + y_test_outsider

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(employee_features)


param_grid = {
    'nu': [0.01, 0.05, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}


# 创建 OneClassSVM 对象
oc_svm = OneClassSVM()
oc_svm.fit(X_train_scaled)


# 在测试数据上进行预测
X_test_scaled = scaler.transform(employee_features + outsider_features)
y_pred = oc_svm.predict(X_test_scaled)
y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred]

# 输出预测结果
print("y_pred",y_pred)
print("y_test",y_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")


