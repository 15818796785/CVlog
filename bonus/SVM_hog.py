import os
import tqdm
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from divide import read_and_divide_data
import train_test_split
# 数据集路径
# dataset_path = 'GeorgiaTechFaces/ConvertGrayscaleprocessedset_1'
# masked_path = 'GeorgiaTechFaces/ConvertGrayScaleMaskprocessedset_1'
dataset_path = '20_GeorgiaTechFaces/dataset/part_1'
masked_path = '20_GeorgiaTechFaces/masked/part_1'
# # 读取未戴口罩的数据并提取HOG特征
# X_train = []
# y_train = []
# for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading unmasked images'):
#     if os.path.isdir(os.path.join(dataset_path, subject_name)):
#         subject_images_dir = os.path.join(dataset_path, subject_name)
#         for img_name in os.listdir(subject_images_dir):
#             if img_name.endswith('.jpg'):
#                 img_path = os.path.join(subject_images_dir, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 img = cv2.resize(img, (150, 150))
#                 hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt = True)
#                 X_train.append(hog_features)
#                 y_train.append(int(subject_name[1:]))

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # 读取戴口罩的数据并提取HOG特征
# X_test = []
# y_test = []
# for subject_name in tqdm.tqdm(os.listdir(masked_path), desc='reading masked images'):
#     if os.path.isdir(os.path.join(masked_path, subject_name)):
#         subject_images_dir = os.path.join(masked_path, subject_name)
#         for img_name in os.listdir(subject_images_dir):
#             if img_name.endswith('.jpg'):
#                 img_path = os.path.join(subject_images_dir, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 img = cv2.resize(img, (150, 150))
#                 hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt = True)
#                 X_test.append(hog_features)
#                 y_test.append(int(subject_name[1:]))

# X_test = np.array(X_test)
# y_test = np.array(y_test)
# X_train, X_test, y_train, y_test = read_and_divide_data(dataset_path, masked_path)
X_train, y_train = train_test_split.train_split(dataset_path)
X_test, y_test = train_test_split.test_split(masked_path)
print("length_of_X_train:{}".format(len(X_train)))
print("length_of_X_test:{}".format(len(X_test)))
# temp_X_train=[]
# temp_X_test=[]
# for img in X_train:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hogfeature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt = True)
#     temp_X_train.append(hogfeature)

# X_train = temp_X_train
# for img in X_test:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hogfeature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt = True)
#     temp_X_test.append(hogfeature)

# X_test = temp_X_test
X_train = np.array(X_train)
X_test = np.array(X_test)
# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("len_X_train[1]:{}".format(len(X_train[1])))
print("len_X_test[1]:{}".format(len(X_test[1])))
print("len_X_train:{}".format(len(X_train)))
print("len_X_test:{}".format(len(X_test)))
print("len_y_train:{}".format(len(y_train)))
print("len_y_test:{}".format(len(y_test)))
svm = SVC(C=0.1, kernel='linear', gamma='auto')
svm.fit(X_train, y_train)
print("here!")
# 使用训练好的模型进行预测
y_pred = svm.predict(X_test)

# 计算并打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 打印分类报告
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 保存混淆矩阵和分类报告
conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(y_test), columns=np.unique(y_test))
conf_matrix_df.to_csv('confusion_matrix_SVM_hog.csv', index=True)

with open('classification_report_SVM_hog.txt', 'w') as f:
    f.write(class_report)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_SVM_hog.png')
plt.show()