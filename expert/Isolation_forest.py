import os

import cv2
from skimage.feature import hog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

processedset_path = "../GeorgiaTechFaces/Maskedset_1"

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
                x_feature = hog(img, orientations=8, pixels_per_cell=(3, 3),
                                cells_per_block=(1, 1), visualize=False)
                X.append(x_feature)

employee_features = X[:450]
outsider_features = X[450:]
y_test_employee = ['Accepted'] * 450
y_test_outsider = ['Rejected'] * 300
y_test = y_test_employee + y_test_outsider

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(employee_features)


clf = IsolationForest(random_state=2).fit(X_train_scaled)

X_test_scaled = scaler.transform(employee_features + outsider_features)
y_pred_test = clf.predict(X_test_scaled)


y_pred = ['Accepted' if x == 1 else 'Rejected' for x in y_pred_test]

# 输出预测结果
print("y_pred",y_pred)
print("y_test",y_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"One class Model accuracy: {accuracy * 100:.2f}%")

