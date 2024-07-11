import os
import random

import cv2
import dlib
import face_recognition
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

dataset_path = "../GeorgiaTechFaces/Crop_1"
masked_dataset_path = "../GeorgiaTechFaces/Maskedcrop_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'

# 加载面部检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 加载图像并组织成结构化数据
X = []
Y = []
X_test = []
Y_test = []
label_index = 0
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='Reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        temp_y_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                # img = face_recognition.load_image_file(img_path)
                # face_locations = face_recognition.face_locations(img)
                # face_encodings = face_recognition.face_encodings(img, face_locations)
                img = cv2.imread(img_path, flags=0)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                img = np.asarray(img, dtype=np.float32)
                #img = np.expand_dims(img, axis=0)


                print(img.shape)
                x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                                cells_per_block=(1, 1), visualize=False)
                X.append(x_feature)
                Y.append(label_index)
    label_index = label_index+1


# 将X分类成employee和outsider
data = list(zip(X, Y))
random.shuffle(data)
X, Y = zip(*data)
# 将数据类型转换为 float32
X = np.array(X)  # 将列表转换为 numpy 数组
Y = np.array(Y)  # 假设 Y 是整数标签数组




# X = [item for sublist in X for item in sublist]
# Y = [item for sublist in Y for item in sublist]

label_index = 0

for subject_name in tqdm.tqdm(os.listdir(masked_dataset_path), desc='Reading masked images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                # img = face_recognition.load_image_file(img_path)
                # face_locations = face_recognition.face_locations(img)
                # face_encodings = face_recognition.face_encodings(img, face_locations)
                img = cv2.imread(img_path, flags=0)
               # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                # 转换为 float32 类型
                # = np.asarray(img, dtype=np.float32)

                # x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                #                 cells_per_block=(1, 1), visualize=False)
                x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                                cells_per_block=(1, 1), visualize=False)
                X_test.append(x_feature)
                Y_test.append(label_index)
    label_index = label_index + 1

# 将X分类成employee和outsider
data = list(zip(X_test, Y_test))
random.shuffle(data)
X_test, Y_test = zip(*data)
# 将数据类型转换为 float32
# X_test = np.array(X_test)  # 将列表转换为 numpy 数组
# Y_test = np.array(Y_test)  # 假设 Y 是整数标签数组

# 创建朴素贝叶斯分类器
model = GaussianNB()

# 训练模型
model.fit(X, Y)

# 预测测试集
Y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print(f"准确率：{accuracy}")