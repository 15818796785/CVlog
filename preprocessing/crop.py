import os
import detector3
import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
# import more libraries as you need
import cv2
import detector2
import tqdm

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure


dataset_path = "../GeorgiaTechFaces/Dataset_1"
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

dataset_path = '../GeorgiaTechFaces/Maskedset_1'
X_masked = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_mask_list = []
        for img_name in os.listdir(subject_images_dir):
            # write code to read each 'img'
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                # add the img to temp_x_list
                temp_x_mask_list.append(img)
        # add the temp_x_list to X
        X_masked.append(temp_x_mask_list)

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
        # gray_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        # gray_mask_image = cv2.cvtColor(temp_maskimg, cv2.COLOR_BGR2GRAY)
        # 计算直方图
        # hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

        # plt.figure()
        # plt.title("Grayscale Histogram")
        # plt.xlabel("Gray level")
        # plt.ylabel("Frequency")
        # plt.plot(hist)
        # plt.xlim([0, 256])
        # plt.show()
        # dlib.hit_enter_to_continue()

        # append the converted image into temp_X_processed
        temp_X_processed.append(temp_img)
        temp_X_maskprocessed.append(temp_maskimg)

    # append temp_X_processed into  X_processed
    X_processed.append(temp_X_processed)
    X_maskprocessed.append(temp_X_maskprocessed)
   
# Save the processed images
maskprocessed_dataset_path = '../GeorgiaTechFaces/Maskedcrop_1'
for i, subject_images in enumerate(X_maskprocessed):
    subject_folder = os.path.join(maskprocessed_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)

processed_dataset_path = '../GeorgiaTechFaces/Crop_1'
for i, subject_images in enumerate(X_processed):
    subject_folder = os.path.join(processed_dataset_path, f"s{str(i + 1).zfill(2)}")
    os.makedirs(subject_folder, exist_ok=True)
    for j, img in enumerate(subject_images):
        cv2.imwrite(os.path.join(subject_folder, f"{str(j + 1).zfill(2)}.jpg"), img)


