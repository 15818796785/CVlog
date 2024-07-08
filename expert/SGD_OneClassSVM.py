import os

import cv2
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

font = {"weight": "normal", "size": 15}

matplotlib.rc("font", **font)

random_state = 42
rng = np.random.RandomState(random_state)

# # Generate train data
# X = 0.3 * rng.randn(500, 2)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * rng.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))


X = []
y = []
X_train = []

processedset_path = "../GeorgiaTechFaces/ConvertGrayscaleMaskprocessedset_1"

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(employee_features)











# OCSVM hyperparameters
nu = 0.05
gamma = 2.0

# Fit the One-Class SVM
clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
clf.fit(X_train_scaled)
y_pred_train = clf.predict(X_train_scaled)

X_test_scaled = scaler.transform(employee_features + outsider_features)
y_pred_test = clf.predict(X_test_scaled)


y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred_test]

# 输出预测结果
print("y_pred",y_pred)
print("y_test",y_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"One class Model accuracy: {accuracy * 100:.2f}%")





# Fit the One-Class SVM using a kernel approximation and SGD
transform = Nystroem(gamma=gamma, random_state=random_state)
clf_sgd = SGDOneClassSVM(
    nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4
)
pipe_sgd = make_pipeline(transform, clf_sgd)
pipe_sgd.fit(X_train_scaled)
y_pred_train_sgd = pipe_sgd.predict(X_train_scaled)
y_pred_test_sgd = pipe_sgd.predict(X_test_scaled)

y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred_test_sgd]

# 输出预测结果
print("y_pred",y_pred)
print("y_test",y_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"One class SGD Model accuracy: {accuracy * 100:.2f}%")


