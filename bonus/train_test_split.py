import os
import tqdm
import cv2
import numpy as np
from skimage.feature import hog

# path = "20_classified_image_washed"



# path = '20_GeorgiaTechFaces/dataset/part_1'

def train_split(path):
    X_train = []
    y_train = []

    for forder in tqdm.tqdm(os.listdir(path), desc='split train data:'):
          for i, img in enumerate(os.listdir(os.path.join(path, forder))):
             if (i < len(os.listdir(os.path.join(path, forder))) / 3):
                 p = os.path.join(path, forder, img)
                 
                 img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                 img = cv2.resize(img, (64, 64))
                 img = np.asarray(img, dtype=np.float32)
                 hog_features = hog(img, orientations=6, pixels_per_cell=(9, 9), cells_per_block=(2, 2), visualize=False,
                               transform_sqrt=True)
                 X_train.append(hog_features)
                 y_train.append(forder)
             else:
                 break

    return X_train, y_train


def test_split(path):
    X_test = []
    y_test = []

    for forder in tqdm.tqdm(os.listdir(path), desc='split test data:'):
        for i, img in enumerate(os.listdir(os.path.join(path, forder))):
            if (i >= len(os.listdir(os.path.join(path, forder)))-len(os.listdir(os.path.join(path, forder))) / 3):
                p = os.path.join(path, forder, img)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                img = np.asarray(img, dtype=np.float32)
                hog_features = hog(img, orientations=6, pixels_per_cell=(9, 9), cells_per_block=(2, 2), visualize=False,
                                   transform_sqrt=True)
                X_test.append(hog_features)
                y_test.append(forder)
            else:
                continue

    return X_test, y_test

# print(train_split(path))