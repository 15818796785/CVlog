import os
import cv2
import dlib
import numpy as np
from skimage.feature import hog
import tqdm
import preprocessing_utils

# 定义路径
dataset_path = "../20_GeorgiaTechFaces/dataset/part_5"
predictor_path = '../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'
Masked_dataset_path = "../20_GeorgiaTechFaces/masked/part_5"
related_dataset_path = "../20_GeorgiaTechFaces/related/part_5"
rotate_dataset_path = "../20_GeorgiaTechFaces/rotate/part_5"

# 初始化dlib的面部检测器和面部标志预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 定义HOG特征提取函数
def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features, hog_image

# 定义添加口罩的函数
def add_mask(image, landmarks):
    # 使用适当的特征点来绘制口罩
    mask_points = landmarks[1:16]  # 选择适当的面部标志点来绘制口罩
    mask_points = np.concatenate([mask_points, [landmarks[35], landmarks[27], landmarks[31]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [mask_points], (255, 255, 255))
    masked_image = cv2.addWeighted(image, 1, mask, 1, 0)
    return masked_image

# 读取图片并处理
X_masked = []
related = []
rotate = []

for person_id in tqdm.tqdm(os.listdir(dataset_path), desc='Processing persons'):
    person_folder = os.path.join(dataset_path, person_id)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    dets = detector(img, 1)
                    if len(dets) == 0:
                        print(f'No features detected in image {img_name}')
                        continue
                    for k, d in enumerate(dets):
                        shape = predictor(img, d)
                        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                        masked_face = add_mask(img, landmarks)
                        rotate_face = preprocessing_utils.random_rotate(img)
                        related.append((person_id, img_name, img))
                        rotate.append((person_id, img_name, rotate_face))
                        X_masked.append((person_id, img_name, masked_face))

# 确保目录存在，如果不存在则创建
os.makedirs(Masked_dataset_path, exist_ok=True)
os.makedirs(related_dataset_path, exist_ok=True)
os.makedirs(rotate_dataset_path, exist_ok=True)

# 保存处理后的图片到相应的子文件夹
for person_id, img_name, img in tqdm.tqdm(X_masked, desc='saving masked images'):
    person_folder = os.path.join(Masked_dataset_path, person_id)
    os.makedirs(person_folder, exist_ok=True)
    cv2.imwrite(os.path.join(person_folder, img_name), img)

for person_id, img_name, img in tqdm.tqdm(related, desc='saving related images'):
    person_folder = os.path.join(related_dataset_path, person_id)
    os.makedirs(person_folder, exist_ok=True)
    cv2.imwrite(os.path.join(person_folder, img_name), img)

for person_id, img_name, img in tqdm.tqdm(rotate, desc='saving rotated images'):
    person_folder = os.path.join(rotate_dataset_path, person_id)
    os.makedirs(person_folder, exist_ok=True)
    cv2.imwrite(os.path.join(person_folder, img_name), img)

print("Processing complete.")
