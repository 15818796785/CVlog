# Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

MAX_SLIDER_VALUE = 255
NUM_EIGEN_FACES = 10


# Create data matrix from a list of images
def createDataMatrix(images):
    print("Creating data matrix", end=" ... ")
    ''' 
    Allocate space for all images in one data matrix.
    The size of the data matrix is
    ( w  * h  * 3, numImages )
    where,
    w = width of an image in the dataset.
    h = height of an image in the dataset.
    3 is for the 3 color channels.
    '''

    numImages = len(images)
    sz = images[0].shape
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i, :] = image

    print("DONE")
    return data


# Read images from the directory


# Add the weighted eigen faces to the mean face
def createNewFace(averageFace,sliderValues,eigenFaces):
    # Start with the mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        '''
        OpenCV does not allow slider values to be negative. 
        So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
        '''
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE / 2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imshow("Result", output)


def resetSliderValues(averageFace,sliderValues,eigenFaces):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE / 2));
    createNewFace(averageFace,sliderValues,eigenFaces)





# 用于识别给定图像的人脸的函数
def recognize_face(face_image, eigenVectors, mean, weights):
    # 将输入图像转换为与训练数据相同的形式
    face_flattened = face_image.flatten().astype(np.float32)
    face_weight = np.dot(face_flattened - mean, eigenVectors.T)

    # 计算与训练集中所有脸的欧式距离
    distances = np.linalg.norm(weights - face_weight, axis=1)
    nearest_face_index = np.argmin(distances)

    # 返回最近的脸的索引和距离
    return nearest_face_index, distances[nearest_face_index]


def test_threshold(X_test, eigenVectors, mean, weights, y_test, thresholds):
    best_accuracy = 0
    best_threshold = None
    results = []

    # 测试不同的阈值以找到最优阈值
    for threshold in thresholds:
        y_pred = []
        for face in X_test:
            _, dist = recognize_face(face, eigenVectors, mean, weights)
            if dist < threshold:
                y_pred.append('Accepted')
            else:
                y_pred.append('Rejected')

        accuracy = accuracy_score(y_test, y_pred)
        results.append((threshold, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy, results


# 用于识别给定图像的人脸的函数
def recognize_specific_face(face_image, eigenVectors, mean, weights, labels):
    # 将输入图像转换为与训练数据相同的形式
    face_flattened = face_image.flatten().astype(np.float32)
    face_weight = np.dot(face_flattened - mean, eigenVectors.T)

    # 计算与训练集中所有脸的欧式距离
    distances = np.linalg.norm(weights - face_weight, axis=1)
    nearest_face_index = np.argmin(distances)

    # 返回最近的脸的索引和距离
    return labels[nearest_face_index], distances[nearest_face_index]
