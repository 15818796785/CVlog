from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import face_recognition
import tqdm
import os
import cv2
import numpy as np
import random
import time

def shuffle_array(X, y):
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    X_shuffled, y_shuffled = zip(*zipped)
    return list(X_shuffled), list(y_shuffled)

processedset_path = "GeorgiaTechFaces/gray_1"
test_path = "GeorgiaTechFaces/Maskedgray_1"

X_processed = []
y = []

for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images'):
    # 取subject_name的后两位数字作为y，并转换为int
    y.append(int(subject_name[-2:]))
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
for subject_name in tqdm.tqdm(os.listdir(test_path), desc='reading test processed images'):
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
# zipped = list(zip(X_readTest, X_processed, y))
# random.shuffle(zipped)
# X_read_shuffled, X_processed_shuffled, y_shuffled = zip(*zipped)
# X_readTest = list(X_read_shuffled)
# X_processed = list(X_processed_shuffled)
# y = list(y_shuffled)


# Train a face recognizer on the Employee set
employee_encodings = []

# for employee_images in X_employee:
for employee_images in tqdm.tqdm(X_processed, desc='employee training'):
    temp_list = []
    for image in employee_images:
        face_locations = face_recognition.face_locations(image)
        employee_encoding = face_recognition.face_encodings(image, face_locations)
        if employee_encoding != []:
            temp_list.append(employee_encoding)
        # use one picture to train for one person
        # break
    # 求数组平均值
    avg = np.mean(temp_list, axis=0)
    # employee_encodings.append(temp_list)
    employee_encodings.append(avg[0])
    # print(avg)
    
# print("employee_encodings_size", len(employee_encodings)) 
# print("employee_encodings_size", len(employee_encodings[0]))
# print("employee_encodings", employee_encodings)

# shuffle the test set
# zipped = list(zip(X_readTest, y))
# random.shuffle(zipped)
# X_shuffled, y_shuffled = zip(*zipped)
# X_test = list(X_shuffled)
# y_test = list(y_shuffled)
X_test = X_readTest
y_test = y
X_test, y_test = shuffle_array(X_test, y_test)

test_sum = 0
probe_succuess_sum = 0
order = 1

for employee_images in X_test:
    print(f"Processing image {order}", end=" ")
    probe_index = y_test[order - 1]
    order += 1
    flag = 0
    test_sum += 1
    # select_num = 0
    select_num = random.randint(0, len(employee_images) - 1)
    image = employee_images[select_num]
    probe_encoding = face_recognition.face_encodings(image)
    start_time = time.time()
    while probe_encoding == [] and select_num < len(employee_images):
        # select_num += 1
        select_num = random.randint(0, len(employee_images) - 1)
        image = employee_images[select_num]
        probe_encoding = face_recognition.face_encodings(image)
        end_time = time.time()
        if end_time - start_time > 5:
            select_num = -1
            break
    print("with index", select_num)
    employee_encodings = np.array(employee_encodings)
    # print(type(employee_encodings), type(employee_encodings[0]), type(employee_encodings[0][0]))
    results = face_recognition.compare_faces(employee_encodings, probe_encoding, tolerance=0.37)
    print("expected:", probe_index, end=" ")
    if results.count(True) == 0:
        print("None result")
    elif results.count(True) > 1:
        print("Multiple result")
    if len(results) and results[probe_index - 1] and results.count(True) == 1:
        print("result:", results.index(True) + 1)
        print("SUCCESS")
        probe_succuess_sum += 1
    else:
        if results.count(True) == 1:
            print("result:", results.index(True) + 1)
        print("FAILED")
    # use break
    # 0.35 62%
    # 0.5 82%
    # not use break
    # 0.5 82%
    # 0.4 94%
    # 0.37 100%
    # 0.35 94%
    # 0.3 86%

# Evaluate the classifier
accuracy = probe_succuess_sum / test_sum
print(f"Accuracy: {accuracy * 100:.2f}%")