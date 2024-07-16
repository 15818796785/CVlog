import os
import detector3
import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
import cv2
import tqdm

# 设置数据集路径
dataset_path = "../20_GeorgiaTechFaces/related/part_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 读取原始数据集
X = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                temp_x_list.append((img_name, img))
        X.append((subject_name, temp_x_list))

# 读取有遮挡的数据集
dataset_path = '../20_GeorgiaTechFaces/masked/part_1'
X_masked = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_mask_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                temp_x_mask_list.append((img_name, img))
        X_masked.append((subject_name, temp_x_mask_list))

# 预处理图像
X_maskprocessed = []
X_processed = []
for i in tqdm.tqdm(range(len(X)), desc='preprocessing images'):
    temp_X_processed = []
    temp_X_maskprocessed = []
    subject_name, x_list = X[i]
    _, x_masklist = X_masked[i]
    for j in range(len(x_list)):
        img_name, img = x_list[j]
        _, mask_img = x_masklist[j]
        temp_img, temp_maskimg = detector3.crop_and_resize_face(img, mask_img, detector, predictor)
        if temp_img is not None:
            temp_X_processed.append((img_name, temp_img))
        if temp_maskimg is not None:
            temp_X_maskprocessed.append((img_name, temp_maskimg))
    X_processed.append((subject_name, temp_X_processed))
    X_maskprocessed.append((subject_name, temp_X_maskprocessed))

# 保存处理后的图像
maskprocessed_dataset_path = '../20_GeorgiaTechFaces/Maskedcrop_1/part_1'
os.makedirs(maskprocessed_dataset_path, exist_ok=True)
for subject_name, subject_images in X_maskprocessed:
    subject_folder = os.path.join(maskprocessed_dataset_path, subject_name)
    os.makedirs(subject_folder, exist_ok=True)
    for img_name, img in subject_images:
        cv2.imwrite(os.path.join(subject_folder, img_name), img)

processed_dataset_path = '../20_GeorgiaTechFaces/Crop_1/part_1'
os.makedirs(processed_dataset_path, exist_ok=True)
for subject_name, subject_images in X_processed:
    subject_folder = os.path.join(processed_dataset_path, subject_name)
    os.makedirs(subject_folder, exist_ok=True)
    for img_name, img in subject_images:
        cv2.imwrite(os.path.join(subject_folder, img_name), img)