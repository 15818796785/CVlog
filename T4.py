from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import tqdm
import os
import cv2
import numpy as np

# 从文件夹中读取数据
maskdataset_path = "GeorgiaTechFaces/Maskedset_1"
processdataset_path = "GeorgiaTechFaces/Processedset_1"


X_masked = []
y = []
for subject_name in tqdm.tqdm(os.listdir(maskdataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(maskdataset_path, subject_name)):
        y.append(subject_name)
        subject_images_dir = os.path.join(maskdataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_x_list.append(gray_img)
        X_masked.append(temp_x_list)

X_processed = []
for subject_name in tqdm.tqdm(os.listdir(processdataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(processdataset_path, subject_name)):
        subject_images_dir = os.path.join(processdataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_x_list.append(gray_img)
        X_processed.append(temp_x_list)


# 将X_processed和X_masked组合起来作为输入特征
X = np.concatenate([X_processed, X_masked], axis=1)

# 分割数据集为训练集和测试集，使用80-20的比例，并且打乱数据
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 训练SVM二分类器
model = SVC(kernel='linear')
model.fit(x_train, y_train)

# 使用测试集评估分类器
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
