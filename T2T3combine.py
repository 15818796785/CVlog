import os

import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
# import more libraries as you need
import cv2
import detector3
import tqdm

dataset_path = "GeorgiaTechFaces/Dataset_1"
predictor_path = 'shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

X = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            # write code to read each 'img'
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                # add the img to temp_x_list
                temp_x_list.append(img)
        # add the temp_x_list to X
        X.append(temp_x_list)

def add_mask(image, landmarks):
    # 使用适当的特征点来绘制口罩
    mask_points = landmarks[1:16]  # 选择适当的面部标志点来绘制口罩
    mask_points = np.concatenate([mask_points, [landmarks[35], landmarks[27], landmarks[31]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [mask_points], (255, 255, 255))
    masked_image = cv2.addWeighted(image, 1, mask, 1, 0)
    return masked_image


X_masked = []

for x_list in tqdm.tqdm(X, desc='adding masks'):
    temp_X_masked = []
    for x in x_list:
        dets = detector(x, 1)
        if len(dets) == 0:
            print('no featrues')
        for k, d in enumerate(dets):
            shape = predictor(x, d)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            masked_face = add_mask(x, landmarks)
            temp_X_masked.append(masked_face)
    X_masked.append(temp_X_masked)

Masked_dataset_path = "GeorgiaTechFaces/Maskedset_1"
# Save the processed images
for i, subject_images in enumerate(X_masked):
    subject_folder = os.path.join(Masked_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)

X_maskprocessed = []
X_processed = []
for i in tqdm.tqdm(range(len(X)), desc='preprocessing images '):
    temp_X_processed = []
    temp_X_maskprocessed = []
    x_list = X[i]
    x_masklist = X_masked[i]
    for j in range(len(x_list)):
        # write the code to detect face in the image (x) using dlib facedetection library
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150

        temp_img, temp_maskimg = detector3.crop_and_resize_face(x_list[j], x_masklist[j], detector, predictor)

        # write the code to convert the image (x) to grayscale
        gray_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        gray_mask_image = cv2.cvtColor(temp_maskimg, cv2.COLOR_BGR2GRAY)
        # 计算直方图
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

        # append the converted image into temp_X_processed
        temp_X_processed.append(gray_image)
        temp_X_maskprocessed.append(gray_mask_image)

    # append temp_X_processed into  X_processed
    X_processed.append(temp_X_processed)
    X_maskprocessed.append(temp_X_maskprocessed)
   
# Save the processed images
maskprocessed_dataset_path = 'GeorgiaTechFaces/Maskprocessedset_1'
for i, subject_images in enumerate(X_maskprocessed):
    subject_folder = os.path.join(maskprocessed_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)

processed_dataset_path = 'GeorgiaTechFaces/Processedset_1'
for i, subject_images in enumerate(X_processed):
    subject_folder = os.path.join(processed_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)
print("finished T3")


