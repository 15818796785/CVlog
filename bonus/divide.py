import os
import tqdm
import cv2
import numpy as np
import random
import face_recognition

random.seed(42)

def shuffle_arrays(arr1, arr2):
    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    return zip(*combined)


def read_and_divide_data(unmasked_path, masked_path):
    X_unmasked = []
    X_masked = []

    # 读取未戴口罩的图像
    for subject_name in tqdm.tqdm(os.listdir(unmasked_path), desc='reading unmasked images'):
        if os.path.isdir(os.path.join(unmasked_path, subject_name)):
            subject_images_dir = os.path.join(unmasked_path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path)
                    temp_x_list.append(img)
            X_unmasked.append(temp_x_list)

    for subject_name in tqdm.tqdm(os.listdir(masked_path), desc='reading masked images'):
        if os.path.isdir(os.path.join(masked_path, subject_name)):
            subject_images_dir = os.path.join(masked_path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path)
                    temp_x_list.append(img)
            X_masked.append(temp_x_list)

    for i in range(len(X_masked)):
        unmasklist = X_unmasked[i]
        masklist = X_masked[i]
        X_unmasked[i], X_masked[i] = shuffle_arrays(unmasklist, masklist)

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for i in range(len(X_unmasked)):
        temp_list = X_unmasked[i]
        len_list = len(temp_list)
        # print("len_list:{}".format(len_list))
        for j in range(len_list):
            
            if j>9:break
            # hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt = True)
            X_train.append(temp_list[j])
            y_train.append(i+1)
    
    for i in range(len(X_masked)):
        temp_list = X_masked[i]
        len_list = len(temp_list)
        for j in range(10,len_list):
            X_test.append(temp_list[j])
            y_test.append(i+1)

    return X_train, X_test, y_train, y_test

# def split_data(X_unmasked, X_masked):
#     X_train = []
#     X_test = []
#     y_train = []
#     y_test = []

#     for i in range(len(X_unmasked)):
#         temp_list = X_unmasked[i]
#         len_list = len(temp_list)
#         for j in range(len_list):
#             if j>9:break
#             X_train.append(temp_list[j])
#             y_train.append(i+1)
    
#     for i in range(len(X_masked)):
#         temp_list = X_masked[i]
#         len_list = len(temp_list)
#         for j in range(10,len_list):
#             X_test.append(temp_list[j])
#             y_test.append(i+1)

#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)
#     return X_train, X_test, y_train, y_test

# unmasked_path = 'GeorgiaTechFaces/ConvertGrayscaleprocessedset_1'
# masked_path = 'GeorgiaTechFaces/ConvertGrayscaleMaskprocessedset_1'

# X_unmasked, X_masked = read_and_shuffle_data(unmasked_path, masked_path)
# X_train, X_test, y_train, y_test = split_data(X_unmasked, X_masked)

