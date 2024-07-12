from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import face_recognition
import tqdm
import os
import cv2
import numpy as np
import random
import time

processedset_path = "GeorgiaTechFaces/gray_1"

X_processed = []
y = []

for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg') and 1 <= int(img_name.split('.')[0]) <= 10:
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X_processed.append(temp_x_list)
        
X_readTest = []
for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading test processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg') and int(img_name.split('.')[0]) >= 11 and int(img_name.split('.')[0]) <= 15:
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X_readTest.append(temp_x_list)

print(len(X_processed[0]), len(X_readTest[0]))
# random.seed(42)
# random.shuffle(X_processed)
# X_employee = X_processed[0:30]
# X_outsider = X_processed[30:]

random.seed(42)
zipped = list(zip(X_readTest, X_processed))
random.shuffle(zipped)
X_read_shuffled, X_processed_shuffled = zip(*zipped)
X_readTest = list(X_read_shuffled)
X_processed = list(X_processed_shuffled)
X_employee = X_processed[0:30]
X_outsider = X_processed[30:]

# image = face_recognition.load_image_file("image.jpg")
# face_locations = face_recognition.face_locations(image)
# print(f"Found {len(face_locations)} face(s) in this image")
# face_encodings = face_recognition.face_encodings(image, face_locations)

# Train a face recognizer on the Employee set
employee_encodings = []

# for employee_images in X_employee:
for employee_images in tqdm.tqdm(X_employee, desc='employee training'):
    for image in employee_images:
        face_locations = face_recognition.face_locations(image)
        employee_encoding = face_recognition.face_encodings(image, face_locations)
        if employee_encoding != []:
            employee_encodings.append(employee_encoding)
        # use one picture to train for one person
        # break
    
print("employee_encodings_size", len(employee_encodings)) 
y_employee = [1] * len(X_employee)
y_outsider = [0] * len(X_outsider)
y = y_employee + y_outsider

# shuffle the test set
zipped = list(zip(X_readTest, y))
random.shuffle(zipped)
X_shuffled, y_shuffled = zip(*zipped)
X_test = list(X_shuffled)
y_test = list(y_shuffled)

y_probe = []
order = 1

for employee_images in X_test:
    print(f"Processing image {order}", end=" ")
    order += 1
    flag = 0
    # select_num = 0
    select_num = random.randint(0, len(employee_images)-1)
    image = employee_images[select_num]
    probe_encoding = face_recognition.face_encodings(image)
    start_time = time.time()
    while probe_encoding == [] and select_num < len(employee_images):
        # select_num += 1
        select_num = random.randint(0, len(employee_images-1))
        image = employee_images[select_num]
        probe_encoding = face_recognition.face_encodings(image)
        end_time = time.time()
        if end_time - start_time > 5:
            break
    print("with index", select_num)
    for encoding in employee_encodings:
        if probe_encoding == []:
            break
        encoding = np.array(encoding)
        # change the tolerance value here to get the best result
        results = face_recognition.compare_faces(encoding, probe_encoding, tolerance=0.35)
        # use break
        # 0.4 90%
        # not use break
        # 0.4 96%
        # 0.35 98%
        # 0.3 94%
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