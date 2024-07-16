import preprocessing_utils
import dlib
import cv2
import os
import tqdm

dataset_path = "../20_GeorgiaTechFaces/crop"
Masked_dataset_path = "../20_GeorgiaTechFaces/mask_crop"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

X = []
for person_id in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    person_folder = os.path.join(dataset_path, person_id)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    mirrored_face = preprocessing_utils.apply_horizontal_flip(img)
                    X.append((person_id, img_name, mirrored_face))

X_masked = []
for person_id in tqdm.tqdm(os.listdir(Masked_dataset_path), desc='reading images'):
    person_folder = os.path.join(Masked_dataset_path, person_id)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    mirrored_face = preprocessing_utils.apply_horizontal_flip(img)
                    X_masked.append((person_id, img_name, mirrored_face))

X_mirror_dataset_path = '../20_GeorgiaTechFaces/mirror'
os.makedirs(X_mirror_dataset_path, exist_ok=True)
# 保存处理后的图片到相应的子文件夹
for person_id, img_name, img in tqdm.tqdm(X, desc='saving mirrored images'):
    person_folder = os.path.join(X_mirror_dataset_path, person_id)
    os.makedirs(person_folder, exist_ok=True)
    cv2.imwrite(os.path.join(person_folder, img_name), img)

X_masked_mirror_dataset_path = '../20_GeorgiaTechFaces/mirror_mask'
os.makedirs(X_masked_mirror_dataset_path, exist_ok=True)
for person_id, img_name, img in tqdm.tqdm(X_masked, desc='saving mirrored images'):
    person_folder = os.path.join(X_masked_mirror_dataset_path, person_id)
    os.makedirs(person_folder, exist_ok=True)
    cv2.imwrite(os.path.join(person_folder, img_name), img)
