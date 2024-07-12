import os
import random

import cv2
import dlib
import face_recognition
import numpy as np
import tqdm
from keras.src.saving import load_model
from keras.src.saving.saving_api import save_model
from matplotlib import pyplot as plt
from skimage.feature import hog

from CVlog.bonus.utils import load_image_dataset, read_200000_Y, shuffle, plot_keras_model

dataset_path = "../img_align_celeba"
masked_dataset_path = "../img_align_celeba"


# 加载图像并组织成结构化数据
X_train = []
#label_index = 0
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='Reading images'):

            if img_name.endswith('.jpg'):
                img_path = os.path.join(dataset_path, img_name)
                # img = face_recognition.load_image_file(img_path)
                # face_locations = face_recognition.face_locations(img)
                # face_encodings = face_recognition.face_encodings(img, face_locations)
                img = cv2.imread(img_path, flags=0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 float32 类型
                img = np.asarray(img, dtype=np.float32)
                #img = np.expand_dims(img, axis=0)


                print(img.shape)
                X_train.append(img)

# 假设文件名为 filename.txt
filename = '../identity_CelebA.txt'
# 打开文件并按行读取
Y_trian = read_200000_Y(filename)

shuffle(X_train,Y_trian)


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

model_path = "../model_ckpts/cnn_200000.h5"

if not os.path.exists(model_path):
    model = model
else:
    model = load_model(model_path)
history = model.fit(X_train, Y_trian, epochs=10, batch_size=32)
save_model(model, model_path)
plot_keras_model(history)




# 在测试集上评估模型
# test_loss, test_acc = model.evaluate(X_test, Y_test)
#print(f'Test accuracy: {test_acc}')
