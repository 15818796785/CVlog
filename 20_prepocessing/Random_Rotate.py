import preprocessing_utils
import dlib
import cv2
import os
import tqdm
import detector2
import detector3


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

dataset_path = "../20_GeorgiaTechFaces/rotate"
rotate = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            rotate.append(img)

X_rotate = []
for x in tqdm.tqdm(X, desc='rotate'):
    rotate_face = preprocessing_utils.random_rotate(x)
    X_rotate.append(rotate_face)

X_masked_rotate = []
for x in tqdm.tqdm(X_masked, desc='rotate'):
    rotate_face = preprocessing_utils.random_rotate(x)
    X_masked_rotate.append(rotate_face)

X_maskprocessed = []
X_processed = []
for i in tqdm.tqdm(range(len(X_rotate)), desc='preprocessing images 1'):
    temp_img = detector2.crop_and_resize_face(X_rotate[i], detector, predictor)
    if temp_img is None:
        continue
    # append the converted image into temp_X_processed
    # append temp_X_processed into  X_processed
    X_processed.append(temp_img)


for i in tqdm.tqdm(range(len(X_masked_rotate)), desc='preprocessing images 2'):
    t, temp_maskimg = detector3.crop_and_resize_face(rotate[i], X_masked_rotate[i], detector, predictor)
    if temp_maskimg is None:
        continue
    # append the converted image into temp_X_processed
    # append temp_X_processed into  X_processed
    X_maskprocessed.append(temp_maskimg)

X_rotate_dataset_path = '../20_GeorgiaTechFaces/rotateprocessedset_1'
if not os.path.exists(X_rotate_dataset_path):
    os.makedirs(X_rotate_dataset_path)
for i, img in enumerate(X_rotate):
    img_save_path = os.path.join(X_rotate_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)

X_masked_rotate_dataset_path = '../20_GeorgiaTechFaces/rotateMaskprocessedset_1'
if not os.path.exists(X_masked_rotate_dataset_path):
    os.makedirs(X_masked_rotate_dataset_path)
for i, img in enumerate(X_masked_rotate):
    img_save_path = os.path.join(X_masked_rotate_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)