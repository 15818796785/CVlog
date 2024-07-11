import os
import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 数据集路径
dataset_path = 'GeorgiaTechFaces/Crop_1'
masked_path = 'GeorgiaTechFaces/Maskedcrop_1'

# 读取未戴口罩的数据
X_train = []
y_train = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading unmasked images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))
                X_train.append(img.flatten())
                y_train.append(int(subject_name[1:]))

X_train = np.array(X_train)
y_train = np.array(y_train)

# 读取戴口罩的数据
X_test = []
y_test = []
for subject_name in tqdm.tqdm(os.listdir(masked_path), desc='reading masked images'):
    if os.path.isdir(os.path.join(masked_path, subject_name)):
        subject_images_dir = os.path.join(masked_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))
                X_test.append(img.flatten())
                y_test.append(int(subject_name[1:]))

X_test = np.array(X_test)
y_test = np.array(y_test)

# 使用SVM进行训练
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = svm.predict(X_test)
print(y_pred)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
