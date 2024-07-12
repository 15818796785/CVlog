import numpy as np
import os
import random
import tqdm
import face_recognition
import dlib
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

dataset_path = "../GeorgiaTechFaces/Maskedset_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'

# 加载面部检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 加载图像并组织成结构化数据
X = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='Reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X.append(temp_x_list)

# 将X分类成employee和outsider
random.shuffle(X)
X = [item for sublist in X for item in sublist]

# 定义员工和外来者的面部图像
employee_faces = X[:30*len(X[0])]
outsider_faces = X[30*len(X[0]):]

def get_face_encodings(face_images):
    encodings = []
    for face in tqdm.tqdm(face_images,desc="extracting face features"):
        face_locations = face_recognition.face_locations(face)
        if face_locations:  # 确保检测到面部
            face_enc = face_recognition.face_encodings(face, face_locations)[0]
            encodings.append(face_enc)
    return encodings

employee_encodings = get_face_encodings(employee_faces)
outsider_encodings = get_face_encodings(outsider_faces)

# 将数据分成训练集和测试集
X_train = np.array(employee_encodings)
X_test = np.array(employee_encodings + outsider_encodings)

# 训练OneClassSVM模型
clf = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
clf.fit(X_train)

# 进行预测
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# 将预测结果转换为标签
y_pred_train_labels = np.where(y_pred_train == 1, "ACCEPT", "REJECTED")
y_pred_test_labels = np.where(y_pred_test == 1, "ACCEPT", "REJECTED")

# 创建真实标签
y_true_test = ["ACCEPT"] * len(employee_encodings) + ["REJECTED"] * len(outsider_encodings)

# 计算准确率
accuracy = accuracy_score(y_true_test, y_pred_test_labels)
print(f"Model accuracy: {accuracy:.2f}")

# # 输出训练和测试的错误数量
# n_error_train = y_pred_train_labels[y_pred_train_labels == "REJECTED"].size
# n_error_test = y_pred_test_labels[y_pred_test_labels == "REJECTED"].size

# print(f"Number of training errors: {n_error_train}")
# print(f"Number of testing errors: {n_error_test}")