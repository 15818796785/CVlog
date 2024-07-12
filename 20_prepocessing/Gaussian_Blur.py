import preprocessing_utils
import dlib
import cv2
import os
import tqdm

dataset_path = "../20_GeorgiaTechFaces/Crop_1"
maskeddataset_path = "../20_GeorgiaTechFaces/Maskedcrop_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

X = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X.append(img)

X_masked = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X_masked.append(img)

X_GaussianBlur = []
for x in tqdm.tqdm(X, desc='gaussian blur'):
    blured_face = preprocessing_utils.apply_gaussian_blur(x)
    X_GaussianBlur.append(blured_face)

X_masked_GaussianBlur = []
for x in tqdm.tqdm(X_masked, desc='gaussian blur'):
    blured_face = preprocessing_utils.apply_gaussian_blur(x)
    X_masked_GaussianBlur.append(blured_face)

X_GaussianBlur_dataset_path = '../20_GeorgiaTechFaces/Gaussianprocessedset_1'
if not os.path.exists(X_GaussianBlur_dataset_path):
    os.makedirs(X_GaussianBlur_dataset_path)
for i, img in enumerate(X_GaussianBlur):
    img_save_path = os.path.join(X_GaussianBlur_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)

X_masked_GaussianBlur_dataset_path = '../20_GeorgiaTechFaces/GaussianMaskprocessedset_1'
if not os.path.exists(X_masked_GaussianBlur_dataset_path):
    os.makedirs(X_masked_GaussianBlur_dataset_path)
for i, img in enumerate(X_masked_GaussianBlur):
    img_save_path = os.path.join(X_masked_GaussianBlur_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)