import preprocessing_utils
import dlib
import cv2
import os
import tqdm

dataset_path = "../20_GeorgiaTechFaces/crop_prolong"
maskeddataset_path = "../20_GeorgiaTechFaces/mask_crop_prolong"
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

X_RandomScale = []
for x in tqdm.tqdm(X, desc='adding masks'):
    scale_face = preprocessing_utils.random_scale(x)
    X_RandomScale.append(scale_face)

X_masked_RandomScale = []
for x in tqdm.tqdm(X_masked, desc='adding masks'):
    scale_face = preprocessing_utils.random_scale(x)
    X_masked_RandomScale.append(scale_face)

X_RandomScale_dataset_path = '../GeorgiaTechFaces/RandomScaleprocessedset_1'
if not os.path.exists(X_RandomScale_dataset_path):
    os.makedirs(X_RandomScale_dataset_path)
for i, img in enumerate(X_RandomScale):
    img_save_path = os.path.join(X_RandomScale_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)

X_masked_RandomScale_dataset_path = '../GeorgiaTechFaces/RandomScaleMaskprocessedset_1'
if not os.path.exists(X_masked_RandomScale_dataset_path):
    os.makedirs(X_masked_RandomScale_dataset_path)
for i, img in enumerate(X_masked_RandomScale):
    img_save_path = os.path.join(X_masked_RandomScale_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)