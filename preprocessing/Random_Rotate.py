import preprocessing_utils
import dlib
import detector3
import detector2
import cv2
import os
import tqdm

dataset_path = "../GeorgiaTechFaces/Dataset_1"
maskeddataset_path = "../GeorgiaTechFaces/Maskedset_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
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

X_masked = []
for subject_name in tqdm.tqdm(os.listdir(maskeddataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(maskeddataset_path, subject_name)):
        subject_images_dir = os.path.join(maskeddataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            # write code to read each 'img'
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                # add the img to temp_x_list
                temp_x_list.append(img)
        # add the temp_x_list to X
        X_masked.append(temp_x_list)

X_RandomRotate = []
for x_list in tqdm.tqdm(X, desc='adding masks'):
    temp_X_blured = []
    for x in x_list:
        rotate_face = preprocessing_utils.random_rotate(x)
        temp_X_blured.append(rotate_face)
    X_RandomRotate.append(temp_X_blured)

X_masked_RandomRotate = []
for x_list in tqdm.tqdm(X_masked, desc='adding masks'):
    temp_X_blured = []
    for x in x_list:
        rotate_face = preprocessing_utils.random_rotate(x)
        temp_X_blured.append(rotate_face)
    X_masked_RandomRotate.append(temp_X_blured)


X_maskprocessed = []
X_processed = []
print(len(X))
for i in tqdm.tqdm(range(len(X)), desc='preprocessing images '):
    temp_X_processed = []
    temp_X_maskprocessed = []
    x_list = X_RandomRotate[i]
    x_masklist = X_masked_RandomRotate[i]
    for j in range(len(x_list)):
        # write the code to detect face in the image (x) using dlib facedetection library
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150

        # temp_img, temp_maskimg = detector3.crop_and_resize_face(x_list[j], x_masklist[j], detector, predictor)
        temp_img = detector2.crop_and_resize_face(x_list[j], detector, predictor)
        temp_maskimg = detector2.crop_and_resize_face(x_masklist[j],detector,predictor)

        # append the converted image into temp_X_processed
        temp_X_processed.append(temp_img)
        temp_X_maskprocessed.append(temp_maskimg)

    # append temp_X_processed into  X_processed
    X_processed.append(temp_X_processed)
    X_maskprocessed.append(temp_X_maskprocessed)

X_RandomRotate_dataset_path = '../GeorgiaTechFaces/RandomRotateprocessedset_1'
for i, subject_images in enumerate(X_processed):
    subject_folder = os.path.join(X_RandomRotate_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        try:
            cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)
        except Exception as e:
            continue

X_masked_RandomRotate_dataset_path = '../GeorgiaTechFaces/RandomRotateMaskprocessedset_1'
for i, subject_images in enumerate(X_maskprocessed):
    subject_folder = os.path.join(X_masked_RandomRotate_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        try:
            cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)
        except Exception as e:
            continue