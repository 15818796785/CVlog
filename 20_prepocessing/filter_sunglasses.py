import os
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import dlib
import numpy as np
from tqdm import tqdm


# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

def is_wearing_sunglasses(image, face, predictor):
    """
    Determine if a person in the image is wearing sunglasses
    """
    landmarks = predictor(image, face)
    left_eye = [36, 37, 38, 39, 40, 41]  # Left eye landmarks
    right_eye = [42, 43, 44, 45, 46, 47]  # Right eye landmarks

    left_eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in left_eye])
    right_eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in right_eye])

    # Extract the ROI of the eye regions
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [left_eye_region], -1, 255, -1)
    cv2.drawContours(mask, [right_eye_region], -1, 255, -1)

    eye_region = cv2.bitwise_and(image, image, mask=mask)
    mean_val = cv2.mean(eye_region, mask=mask)[0]

    # Assume low brightness indicates the person is wearing sunglasses
    return mean_val < 20

def filter_images_with_sunglasses(dataset_path, output_path):
    """
    Filter out images where the person is wearing sunglasses
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(dataset_path):
        for dir_name in tqdm(dirs, desc='processing'):
            dir_path = os.path.join(root, dir_name)
            output_dir_path = os.path.join(output_path, dir_name)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            for file_name in os.listdir(dir_path):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    file_path = os.path.join(dir_path, file_name)
                    image = cv2.imread(file_path)
                    if image is None:
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    sunglasses_detected = False
                    for face in faces:
                        if is_wearing_sunglasses(gray, face, predictor):
                            sunglasses_detected = True
                            break

                    if not sunglasses_detected:
                        output_file_path = os.path.join(output_dir_path, file_name)
                        shutil.copy(file_path, output_file_path)
                    else:
                        print(file_name)
                        cv2.imshow(file_name, image)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()


# Usage example
dataset_path = '../20_GeorgiaTechFaces/related'
output_path = '../20_GeorgiaTechFaces/washed_data'
filter_images_with_sunglasses(dataset_path, output_path)