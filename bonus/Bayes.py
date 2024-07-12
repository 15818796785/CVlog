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

from CVlog.bonus.utils import load_image_dataset, shuffle, save_model, load_model, test_model

dataset_path = "../GeorgiaTechFaces/Dataset_1"
masked_dataset_path = "../GeorgiaTechFaces/Maskedset_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
model_path = "../model_ckpts/gaussian_nb_model.pkl"

# 加载面部检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 加载图像并组织成结构化数据
X = []
Y = []
X_test = []
Y_test = []
X = load_image_dataset(dataset_path)
label_index = 0

for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='Reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                Y.append(label_index)
    label_index = label_index + 1

X, Y = shuffle(X, Y)

X_test = load_image_dataset(masked_dataset_path)
label_index = 0
for subject_name in tqdm.tqdm(os.listdir(masked_dataset_path), desc='Reading masked images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                Y_test.append(label_index)
    label_index = label_index + 1

shuffle(X_test, Y_test)

# 创建朴素贝叶斯分类器
if not os.path.exists(model_path):
    model = GaussianNB()
    model.fit(X, Y)
    save_model(model, model_path)
else:
    model = load_model(model_path)
    # 训练模型

saved_path = "../model_ckpts/"
test_model(model, X_test, Y_test,saved_path,"Bayes_50_Datasets.txt")



