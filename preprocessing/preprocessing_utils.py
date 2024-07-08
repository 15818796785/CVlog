import dlib
import glob
import os
from dlib import get_frontal_face_detector, shape_predictor, image_window
import cv2
from matplotlib import pyplot as plt


def crop_and_resize_face(img, img_masked, detector, predictor):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to dlib compatible color space

        # Detect faces
        dets = detector(img_rgb, 1)
        if len(dets) == 0:
            print('no faces detected')
            return None  # No faces detected

        # Process the first detected face
        d = dets[0]  # Assuming the first detected face is the subject

        shape = predictor(img_rgb, d)

        # Get the bounding box coordinates from the landmarks
        x_min = min([shape.part(i).x for i in range(shape.num_parts)])
        x_max = max([shape.part(i).x for i in range(shape.num_parts)])
        y_min = min([shape.part(i).y for i in range(shape.num_parts)])
        y_max = max([shape.part(i).y for i in range(shape.num_parts)])

        # Crop and resize
        img_cropped_face = img[y_min:y_max, x_min:x_max]
        img_masked_cropped_face = img_masked[y_min:y_max, x_min:x_max]
        img_resized_face = cv2.resize(img_cropped_face, (150, 150))
        img_masked_resized_face = cv2.resize(img_masked_cropped_face, (150, 150))

        # 显示检测到的人脸


        return img_resized_face, img_masked_resized_face



# 使用方法
import cv2
import numpy as np


#高斯模糊
def apply_gaussian_blur(image, kernel_size=(15, 15), sigma=4):
    return cv2.GaussianBlur(image, kernel_size, sigma)


#转换颜色空间
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


#边缘检测
def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

#数据增强
def random_rotate(image):
    angle = np.random.uniform(-30, 30)  # 随机角度
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def random_scale(image):
    fx = fy = np.random.uniform(0.1, 3.0)  # 随机缩放比例
    return cv2.resize(image, None, fx=fx, fy=fy)


