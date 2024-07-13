import numpy as np
import cv2
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from tqdm import tqdm


def load_images(path):
    images = []
    labels = []

    for subject_name in tqdm(os.listdir(path), desc='Reading images'):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            for img_name in os.listdir(subject_images_dir):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Optionally resize the image
                img = cv2.resize(img, (150, 150))
                images.append(img)
                labels.append(int(img_name.split('.')[0]))

    return np.array(images), np.array(labels)


def perform_k_fold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_read_train, X_read_test = shuffle_array(X_train, X_test)
        X_employee = X_read_train[0:30]
        X_outsider = X_read_train[30:]

        y_employee = [1] * len(X_employee)
        y_outsider = [0] * len(X_outsider)
        y = y_employee + y_outsider
        X_test, y_test = shuffle_array(X_read_test, y)
        return X_employee, X_test, y_test

        # 创建模型
        model = SVC()

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上评估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # 打印每一折的训练和测试数据大小
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Accuracy: {accuracy}")

    average_accuracy = np.mean(accuracies)
    print(f"Average Accuracy: {average_accuracy:.2f}")
# Example usage
path = "your/dataset/path"
X, y = load_images(path)
perform_k_fold(X, y, k=5)
