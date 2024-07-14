import os
import detector2
import detector3
import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
# import more libraries as you need
import cv2
import tqdm

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure


dataset_path = "../20_GeorgiaTechFaces/dataset/part_1"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 读取图像编号和人物序号的映射文件
label_mapping_file = "../identity_CelebA.txt"  # 替换为你的映射文件路径
image_labels_dict = {}

with open(label_mapping_file, 'r') as f:
    for line in f:
        img_name, label = line.strip().split()
        image_labels_dict[img_name] = int(label)

X = []
image_numbers = []
count = 0
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        count += 1
        # if count <= 100000:
        #     continue
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        # if len(X) >= 100000:
        #     break
        if img is not None:
            X.append(img)
            image_numbers.append(img_name.split('.')[0] + ".jpg")  # 记录图片编号

dataset_path = '../20_GeorgiaTechFaces/masked/part_1'
X_masked = []
image_numbers_mask = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        # if len(X_masked) >= 5000:
        #     break
        if img is not None:
            X_masked.append(img)
            image_numbers_mask.append(img_name.split('.')[0] + ".jpg")  # 记录图片编号

dataset_path = '../20_GeorgiaTechFaces/related/part_1'
related = []
for img_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if img_name.endswith('.png'):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        # if len(related) >= 5000:
        #     break
        if img is not None:
            related.append(img)



X_maskprocessed = []
X_processed = []
image_labels = []
for i, img_number in tqdm.tqdm(zip(range(len(X)), image_numbers), desc='preprocessing images 1'):
    temp_img = detector2.crop_and_resize_face(X[i], detector, predictor)
    if temp_img is None:
        continue
    # append the converted image into temp_X_processed
    # append temp_X_processed into  X_processed
    X_processed.append(temp_img)
    if img_number in image_labels_dict:
        image_labels.append((img_number, image_labels_dict[img_number]))  # 记录图片编号和人物序号
    else:
        image_labels.append((img_number, -1))


for i in tqdm.tqdm(range(len(related)), desc='preprocessing images 2'):
    t, temp_maskimg = detector3.crop_and_resize_face(related[i], X_masked[i], detector, predictor)
    if temp_maskimg is None:
        continue
        print(i)
        plt.imshow(cv2.cvtColor(related[i], cv2.COLOR_BGR2RGB))  # Display the image with no faces detected
        plt.title('No faces detected')
        plt.show()

    # append the converted image into temp_X_processed
    # append temp_X_processed into  X_processed
    X_maskprocessed.append(temp_maskimg)

# Save the processed images
maskprocessed_dataset_path = '../20_GeorgiaTechFaces/Maskedcrop_1/part_1'
if not os.path.exists(maskprocessed_dataset_path):
    os.makedirs(maskprocessed_dataset_path)
for i, img in enumerate(X_maskprocessed):
    img_save_path = os.path.join(maskprocessed_dataset_path, f"{str(i + 1).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)

processed_dataset_path = '../20_GeorgiaTechFaces/Crop_1/part_1'
if not os.path.exists(processed_dataset_path):
    os.makedirs(processed_dataset_path)
for i, img in enumerate(X_processed):
    img_save_path = os.path.join(processed_dataset_path, f"{str(i + 100001).zfill(6)}.jpg")
    cv2.imwrite(img_save_path, img)


