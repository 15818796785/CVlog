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

processedset_path = "../GeorgiaTechFaces/Maskedset_1"

X_processed = []
y = []

# image = face_recognition.load_image_file("image.jpg")
# face_locations = face_recognition.face_locations(image)
# print(f"Found {len(face_locations)} face(s) in this image")
# face_encodings = face_recognition.face_encodings(image, face_locations)

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

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.75, shuffle=True, random_state=42)
print(f"Training set size: {len(y_train)}", f"Test set size: {len(y_test)}")
y_probe = []

order = 1

for employee_images in X_test:
    print(f"Processing image {order}")
    order += 1
    flag = 0
    image = employee_images[0]
    probe_encoding = face_recognition.face_encodings(image)
    for encoding in employee_encodings:
        encoding = np.array(encoding)
        results = face_recognition.compare_faces(encoding, probe_encoding)
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

# model = SVC(kernel='linear', probability=True)
# model.fit(employee_encodings, ['ACCEPT'] * len(employee_encodings))

# # Evaluate the model on the Employee and Outsider sets
# total_images = 0
# correct_predictions = 0
# for images in X_employee + X_outsider:
#     for image in images:
#         total_images += 1
#         probe_encoding = face_recognition.face_encodings(image)[0]
#         prediction = model.predict([probe_encoding])[0]
#         if prediction == 'ACCEPT' and images in X_employee:
#             correct_predictions += 1
#         elif prediction == 'REJECTED' and images in X_outsider:
#             correct_predictions += 1

# accuracy = correct_predictions / total_images
# print(f'Accuracy: {accuracy:.2f}')