import dlib
import glob
import os
from dlib import get_frontal_face_detector, shape_predictor, image_window
import cv2
from matplotlib import pyplot as plt


def crop_and_resize_face(img, detector, predictor):
        """
        Detect the face, extract facial landmarks, crop the face region, and resize the image.

        Parameters:
        img (numpy.array): The image from which to detect the face and extract landmarks.
        detector (dlib.fhog_object_detector): A dlib face detector.
        predictor (dlib.shape_predictor): A dlib shape predictor.

        Returns:
        numpy.array: The cropped and resized face image.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to dlib compatible color space

        # Detect faces
        dets = detector(img_rgb, 1)
        if len(dets) == 0:
            print('no faces detected')
            return None  # No faces detected

        # Process the first detected face
        d = dets[0]  # Assuming the first detected face is the subject
        # # Extract the face region from the image
        # face_region = img[d.top():d.bottom(), d.left():d.right()]
        #
        # # Convert the face region to a format suitable for displaying
        # # This might involve converting from BGR to RGB if using OpenCV to load images
        # if face_region.ndim == 3:  # Color image
        #         face_region = face_region[:, :, ::-1]  # Converting BGR (OpenCV format) to RGB
        # plt.imshow(face_region)  # 'cmap' is not necessary unless you want a specific colormap
        # plt.colorbar()  # Optionally add a color bar
        # plt.show()

        shape = predictor(img_rgb, d)

        # Get the bounding box coordinates from the landmarks
        x_min = min([shape.part(i).x for i in range(shape.num_parts)])
        x_max = max([shape.part(i).x for i in range(shape.num_parts)])
        y_min = min([shape.part(i).y for i in range(shape.num_parts)])
        y_max = max([shape.part(i).y for i in range(shape.num_parts)])

        # Crop and resize
        cropped_face = img[y_min:y_max, x_min:x_max]
        resized_face = cv2.resize(cropped_face, (150, 150))

        # 显示检测到的人脸


        return resized_face



# 使用方法
import cv2
import numpy as np


#高斯模糊
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
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
    fx = fy = np.random.uniform(0.8, 1.2)  # 随机缩放比例
    return cv2.resize(image, None, fx=fx, fy=fy)


