import os
import pickle
import random

import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.metrics import accuracy_score


# 保存模型
def save_model(model,path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

# 加载模型

def load_model(path):
    with open(path, 'rb') as file:
        loaded_model = pickle.load(file)
        return loaded_model


## use opencv and hog
def load_image_dataset(dataset_path):
    X = []
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
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 转换为 float32 类型
                    img = np.asarray(img, dtype=np.float32)
                    # img = np.expand_dims(img, axis=0)

                    # print(img.shape)
                    x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                                    cells_per_block=(1, 1), visualize=False)
                    X.append(x_feature)
        else:
            if subject_name.endswith('.jpg'):
                img_path = os.path.join(dataset_path, subject_name)
                # img = face_recognition.load_image_file(img_path)
                # face_locations = face_recognition.face_locations(img)
                # face_encodings = face_recognition.face_encodings(img, face_locations)
                img = cv2.imread(img_path, flags=0)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                img = np.asarray(img, dtype=np.float32)
                # img = np.expand_dims(img, axis=0)
                # print(img.shape)
                x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                                cells_per_block=(1, 1), visualize=False)
                X.append(x_feature)
    return X


def shuffle(X,Y):
    # 将X分类成employee和outsider
    data = list(zip(X, Y))
    random.shuffle(data)
    X, Y = zip(*data)
    # 将数据类型转换为 float32
    X = np.array(X)  # 将列表转换为 numpy 数组
    Y = np.array(Y)  # 假设 Y 是整数标签数组
    return X, Y

def test_model(model,X_test,Y_test,saved_path,name):

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(Y_test)
    print(Y_pred)
    # 将准确率输出保存到文件中
    path = os.path.join(saved_path, name)
    with open(path, 'w') as f:
        f.write(f"准确率：{accuracy}\n")
        print(f"准确率：{accuracy}\n")
        print("准确率已保存到文件中。")


def train_model_in_batch(model, X_train, Y_train,current_epoch, batch_size=32, epochs=5):
    """
    使用分批次训练模型的方法。

    Parameters:
    - model: 要训练的模型对象，例如 sklearn 的分类器或者回归器
    - X_train: 训练数据特征集，numpy 数组或类似结构
    - Y_train: 训练数据标签集，numpy 数组或类似结构
    - batch_size: 每个批次的样本数，默认为32
    - epochs: 训练的轮数，默认为5

    Returns:
    - model: 训练完成的模型对象
    """
    num_samples = len(X_train)
    num_batches = num_samples // batch_size

    for epoch in range(current_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]

            # 在这里训练模型
            model.partial_fit(X_batch, Y_batch, classes=np.unique(Y_train))

        # 训练完一个 epoch 后可以输出一些信息，比如损失值或准确率
        # 这里可以根据需要自定义输出内容
        print(f"Epoch {epoch + 1} completed.")

    print("训练完成。")
    return model

def read_200000_Y(filename):
    Y = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 遍历每一行并提取末尾的数字
    for line in lines:
        # 使用 split() 函数分割每一行，默认按空格分割
        parts = line.split()
        # 提取末尾的数字（假设末尾的数字在每行的最后一个位置）
        number = int(parts[-1])
        # 可以在这里进行你的操作，比如打印数字
        Y.append(number)
    return Y

def plot_keras_model(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
