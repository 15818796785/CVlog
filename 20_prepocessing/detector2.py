import dlib
import glob
import os
from dlib import get_frontal_face_detector, shape_predictor, image_window
import cv2
from matplotlib import pyplot as plt


def crop_and_resize_face(img, detector, predictor):
        if img is None:
                return None
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

        if x_min >= x_max or y_min >= y_max:
                print("invalid cropping")
                return None

        # Crop and resize
        cropped_face = img[y_min:y_max, x_min:x_max]
        if cropped_face.size == 0 or cropped_face is None:
                plt.imshow(img)  # Display the image with no faces detected
                plt.title('No faces detected')
                plt.show()
                return None
        resized_face = cv2.resize(cropped_face, (150, 150))

        # 显示检测到的人脸


        return resized_face



# 使用方法
