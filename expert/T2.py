from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import face_recognition
import tqdm
import os
import cv2
import numpy as np
import random

processedset_path = "../GeorgiaTechFaces/Processedset_1"

X_processed = []
y = []
        
for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X_processed.append(temp_x_list)

random.seed(42)
random.shuffle(X_processed)
X_employee = X_processed[0:30]
X_outsider = X_processed[30:]


# Train a face recognizer on the Employee set
employee_encodings = []
# for employee_images in X_employee:
for employee_images in tqdm.tqdm(X_employee, desc='employee training'):
    for image in employee_images:
        employee_encoding = face_recognition.face_encodings(image)
        employee_encodings.append(employee_encoding)
        # use one picture to train for one person
        # break

y_employee = [1] * len(X_employee)
y_outsider = [0] * len(X_outsider)
y = y_employee + y_outsider

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.8, shuffle=True, random_state=42)
print(f"Training set size: {len(y_train)}", f"Test set size: {len(y_test)}")
y_probe = []

order = 1

for employee_images in X_test:
    print(f"Processing image {order}")
    order += 1
    flag = 0
    # select_num = 0
    select_num = random.randint(10, len(employee_images) - 1)
    image = employee_images[select_num]
    probe_encoding = face_recognition.face_encodings(image)
    while probe_encoding == [] and select_num < len(employee_images):
        # select_num += 1
        select_num = random.randint(10, len(employee_images) - 1)
        image = employee_images[select_num]
        probe_encoding = face_recognition.face_encodings(image)
    for encoding in employee_encodings:
        encoding = np.array(encoding)
        # change the tolerance value here to get the best result
        results = face_recognition.compare_faces(encoding, probe_encoding, tolerance=0.5)
        # print(results)
        if len(results) and results[0]:
            if flag == 0:
                print("ACCEPT")
                y_probe.append(1)
                flag = 1
        else:
            pass
    if flag:
        continue
    print("REJECTED")
    y_probe.append(0)

print("y_probe", y_probe)
print("y_test", y_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_probe)
print(f"Accuracy: {accuracy * 100:.2f}%")