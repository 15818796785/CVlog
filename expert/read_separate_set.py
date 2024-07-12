import tqdm
import os
import cv2
import numpy as np
import random
random.seed(42)

def shuffle_array(X, y):
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    X_shuffled, y_shuffled = zip(*zipped)
    return list(X_shuffled), list(y_shuffled)

# return: X_employee(len=30, len[0]=10), X_test(len=50, len[0]=5), y_test(len=50, len[0]=5)
def read_separate_set(path):
    X_read_train = []
    X_read_test = []
    # first 10 img in folder is for training, the rest is for testing
    for subject_name in tqdm.tqdm(os.listdir(path), desc='reading processed images'):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg') and 1 <= int(img_name.split('.')[0]) <= 10:
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path)
                    #cv2.resize(img, (150, 150))
                    temp_x_list.append(img)
            X_read_train.append(temp_x_list)

    for subject_name in tqdm.tqdm(os.listdir(path), desc='reading test processed images'):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            temp_x_list = []
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg') and int(img_name.split('.')[0]) >= 11 and int(img_name.split('.')[0]) <= 15:
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path)
                    #cv2.resize(img, (150, 150))
                    temp_x_list.append(img)
            X_read_test.append(temp_x_list)
    
    # shuffle reading sets
    X_read_train, X_read_test = shuffle_array(X_read_train, X_read_test)
    X_employee = X_read_train[0:30]
    X_outsider = X_read_train[30:]
    
    y_employee = [1] * len(X_employee)
    y_outsider = [0] * len(X_outsider)
    y = y_employee + y_outsider
    X_test, y_test = shuffle_array(X_read_test, y)
    return X_employee, X_test, y_test
            