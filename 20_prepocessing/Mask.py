import os

import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
# import more libraries as you need
import cv2
import tqdm

dataset_path = "img_align_celeba"
predictor_path = 'shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

X = []
count = 0
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X.append(img)
        if len(X) >= 20000:  # 达到20000张图片时结束
            break
# add the temp_x_list to X

def add_mask(image, landmarks):
    # 使用适当的特征点来绘制口罩
    mask_points = landmarks[1:16]  # 选择适当的面部标志点来绘制口罩
    mask_points = np.concatenate([mask_points, [landmarks[35], landmarks[27], landmarks[31]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [mask_points], (255, 255, 255))
    masked_image = cv2.addWeighted(image, 1, mask, 1, 0)
    return masked_image


X_masked = []

for x in tqdm.tqdm(X, desc='adding masks'):
    if len(X_masked) > 100: break
    dets = detector(x, 1)
    if len(dets) == 0:
        print('no featrues')
    for k, d in enumerate(dets):
        shape = predictor(x, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        masked_face = add_mask(x, landmarks)
        X_masked.append(masked_face)

Masked_dataset_path = "20_GeorgiaTechFaces/masked"

# 确保目录存在，如果不存在则创建
if not os.path.exists(Masked_dataset_path):
    os.makedirs(Masked_dataset_path)
# Save the processed images
for i, img in enumerate(X_masked):
    cv2.imwrite(os.path.join(Masked_dataset_path, f"{str(i + 1).zfill(2)}.jpg"), img)