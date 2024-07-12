import os
import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from CVlog.bonus.utils import train_model_in_batch, save_model, load_model, read_200000_Y

# 数据集路径
dataset_path = "../img_align_celeba"
#masked_path = 'GeorgiaTechFaces/Maskedcrop_1'

# 读取未戴口罩的数据
X_train = []
y_train = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading unmasked images'):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(dataset_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))
                img = np.asarray(img, dtype=np.float32)
                X_train.append(img.flatten())
filename = '../identity_CelebA.txt'
y_train = read_200000_Y(filename)
# 转换成 numpy 数组

X_train = np.array(X_train)
y_train = np.array(y_train)

##### Added by xtl
model_path = "../models/SVM_model.pkl"
if not os.path.exists(model_path):
    model = SVC(kernel='linear')
else:
    model = load_model(model_path)
model.fit(X_train, y_train)
save_model(model, model_path)
