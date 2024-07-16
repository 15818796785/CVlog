import dlib
import glob
import os
from dlib import get_frontal_face_detector, shape_predictor, image_window
import cv2
from matplotlib import pyplot as plt


def crop_and_resize_face(img, img_masked, detector, predictor):
    """
    Detect the face, extract facial landmarks, crop the face region, and resize the image.

    Parameters:
    img (numpy.array): The image from which to detect the face and extract landmarks.
    img_masked (numpy.array): The masked image corresponding to img.
    detector (dlib.fhog_object_detector): A dlib face detector.
    predictor (dlib.shape_predictor): A dlib shape predictor.

    Returns:
    numpy.array, numpy.array: The cropped and resized face image and the corresponding masked image.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to dlib compatible color space

    # Detect faces
    dets = detector(img_rgb, 1)
    if len(dets) == 0:
        print('No faces detected')
        return None, None  # No faces detected

    # Process the first detected face
    d = dets[0]  # Assuming the first detected face is the subject

    shape = predictor(img_rgb, d)

    # Get the bounding box coordinates from the landmarks
    x_min = min([shape.part(i).x for i in range(shape.num_parts)])
    x_max = max([shape.part(i).x for i in range(shape.num_parts)])
    y_min = min([shape.part(i).y for i in range(shape.num_parts)])
    y_max = max([shape.part(i).y for i in range(shape.num_parts)])

    # Crop to the upper half of the face (from the top of the face to the nose)
    nose_y = shape.part(30).y  # Nose tip y-coordinate (landmark index 30)

    cropped_face = img[y_min:nose_y, x_min:x_max]
    cropped_maskedface = img_masked[y_min:nose_y, x_min:x_max]

    if cropped_face.size == 0 or cropped_face is None:
        plt.imshow(img)  # Display the image with no faces detected
        plt.title('No faces detected')
        plt.show()
        return None, None
    if cropped_maskedface.size == 0 or cropped_maskedface is None:
        plt.imshow(img)  # Display the image with no faces detected
        plt.title('No faces detected')
        plt.show()

    # Resize
    resized_face = cv2.resize(cropped_face, (150, 150))
    resized_maskedface = cv2.resize(cropped_maskedface, (150, 150))

    return resized_face, resized_maskedface
