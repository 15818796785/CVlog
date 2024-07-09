import os
import random

import cv2
import dlib
import face_recognition
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.feature import hog

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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                img = np.asarray(img, dtype=np.float32)
                #img = np.expand_dims(img, axis=0)


                print(img.shape)
                X.append(img)
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                # 转换为 float32 类型
                img = np.asarray(img, dtype=np.float32)

                # x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                #                 cells_per_block=(1, 1), visualize=False)
                X_test.append(img)
                Y_test.append(label_index)
    label_index = label_index + 1

# 将X分类成employee和outsider
data = list(zip(X_test, Y_test))
random.shuffle(data)
X_test, Y_test = zip(*data)
# 将数据类型转换为 float32
X_test = np.array(X_test)  # 将列表转换为 numpy 数组
Y_test = np.array(Y_test)  # 假设 Y 是整数标签数组


# X_test = [item for sublist in X_test for item in sublist]
# Y_test = [item for sublist in Y_test for item in sublist]


from keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


#定义卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 假设 X_train, y_train, X_test, y_test 是已经准备好的训练和测试数据
history = model.fit(X, Y, epochs=10, batch_size=1, validation_data=(X_test, Y_test))
# 绘制训练过程中的损失值变化
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(X_test, Y_test)
#print(f'Test accuracy: {test_acc}')
