import numpy as np
import os
import random
import tqdm
import face_recognition
import dlib
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from read_separate_set_face_recognition import read_separate_set

dataset_path = "GeorgiaTechFaces/Gray_1"
predictor_path = 'shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'

# 加载面部检测器
detector = dlib.get_frontal_face_detector()


X_employee, X_test, y_test = read_separate_set(dataset_path)

def get_face_encodings(face_images):
    encodings = []
    for face_list in tqdm.tqdm(face_images,desc="extracting face features"):
        temp_list = []
        for face in face_list:
            face_locations = face_recognition.face_locations(face)
            if face_locations:  # 确保检测到面部
                face_enc = face_recognition.face_encodings(face, face_locations)[0]
                temp_list.append(face_enc)
        encodings.append(temp_list)
    return encodings

employee_encodings = get_face_encodings(X_employee)
test_encodings = get_face_encodings(X_test)
temp_y = []
for i in range(len(y_test)):
    len_list = len(test_encodings[i])
    for j in range(len_list):
        temp_y.append(y_test[i])
y_test = temp_y
test_encodings = [item for sublist in test_encodings for item in sublist]
employee_encodings = [item for sublist in employee_encodings for item in sublist]

# 将数据分成训练集和测试集
print("employee_encodings:{}".format(len(employee_encodings)))
# print("outsider_encodings{}".format(len(outsider_encodings)))
X_train = np.array(employee_encodings)
X_test = np.array(test_encodings)

# 训练OneClassSVM模型
# 定义 nu 的不同取值
nus = [0.005,0.01, 0.02, 0.04, 0.06]

# 保存每个 nu 下的准确率
accuracies = []

# 遍历每个 nu 进行训练和测试
for nu in nus:
    clf = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    clf.fit(X_train)

    # 进行预测
    y_pred_test = clf.predict(X_test)
    print("y_pred_test:{}".format(len(y_pred_test)))
    # 将预测结果转换为标签
    y_pred_test_labels = np.where(y_pred_test == 1, "ACCEPT", "REJECTED")

    # 创建真实标签
    # y_true_test = ["ACCEPT"] * len(employee_encodings) + ["REJECTED"] * len(outsider_encodings)
    y_true_test = ["ACCEPT" if i ==1 else "REJECTED" for i in y_test ]
    # 计算准确率
    accuracy = accuracy_score(y_true_test, y_pred_test_labels)
    accuracies.append(accuracy)
    print(f"nu = {nu}, Model accuracy: {accuracy:.2f}")

# 绘制 nu 和准确率的关系图
plt.figure(figsize=(10, 6))
plt.plot(nus, accuracies, marker='o')
plt.title('Model Accuracy for Different nu Values')
plt.xlabel('nu')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('One class svm accuracy_Gray_1.png')
plt.show()

# # 输出训练和测试的错误数量
# n_error_train = y_pred_train_labels[y_pred_train_labels == "REJECTED"].size
# n_error_test = y_pred_test_labels[y_pred_test_labels == "REJECTED"].size

# print(f"Number of training errors: {n_error_train}")
# print(f"Number of testing errors: {n_error_test}")