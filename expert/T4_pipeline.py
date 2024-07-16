from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import tqdm
import os
import cv2
import numpy as np
import face_recognition
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plots the confusion matrix.
    
    Parameters:
    y_true (numpy.ndarray): True labels
    y_pred (numpy.ndarray): Predicted labels
    class_labels (list): List of class labels
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Add values to the grid cells
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def split_and_shuffle_lists(list1, list2, list3, list4, test_size=0.2, random_state=42):
    """
    Shuffles and splits four lists into training and test sets.
    
    Parameters:
    list1 (list): The first list to be split and shuffled.
    list2 (list): The second list to be split and shuffled.
    list3 (list): The third list to be split and shuffled.
    list4 (list): The fourth list to be split and shuffled.
    test_size (float): The proportion of the dataset to include in the test set.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    list1_train, list1_test, list2_train, list2_test, list3_train, list3_test, list4_train, list4_test
    """
    # Combine the lists into a single list of tuples
    combined = list(zip(list1, list2, list3, list4))
    
    # Set the random seed
    random.seed(random_state)
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Calculate the split indices
    total_length = len(combined)
    test_length = int(total_length * test_size)
    train_length = total_length - test_length
    
    # Split the shuffled list into training and test sets
    train_set = combined[:train_length]
    test_set = combined[train_length:]
    
    # Unzip the split sets back into separate lists
    list1_train, list2_train, list3_train, list4_train = zip(*train_set)
    list1_test, list2_test, list3_test, list4_test = zip(*test_set)
    
    return (list(list1_train), list(list1_test),
            list(list2_train), list(list2_test),
            list(list3_train), list(list3_test),
            list(list4_train), list(list4_test))

maskedset_path = "GeorgiaTechFaces/mask_crop_prolong"
processedset_path = "GeorgiaTechFaces/crop_prolong"

X_masked = []
X_processed = []

for subject_name in tqdm.tqdm(os.listdir(maskedset_path), desc='reading masked images'):
    if os.path.isdir(os.path.join(maskedset_path, subject_name)):
        # y.append(subject_name)
        subject_images_dir = os.path.join(maskedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, flags=0)
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_x_list.append(img)
        X_masked.append(temp_x_list)
        
for subject_name in tqdm.tqdm(os.listdir(processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path, flags=0)
                # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_x_list.append(img)
        X_processed.append(temp_x_list)

mask_path = "GeorgiaTechFaces/Maskedgray_1"
X_total = []
for subject_name in tqdm.tqdm(os.listdir(mask_path), desc='reading masked processed images by face-recognition'):
    if os.path.isdir(os.path.join(mask_path, subject_name)):
        subject_images_dir = os.path.join(mask_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            img_path = os.path.join(subject_images_dir, img_name)
            img = face_recognition.load_image_file(img_path)
            temp_x_list.append(img)
        X_total.append(temp_x_list)

unmask_path = "GeorgiaTechFaces/gray_1"
for subject_name in tqdm.tqdm(os.listdir(unmask_path), desc='reading unmasked processed images by face-recognition'):
    if os.path.isdir(os.path.join(unmask_path, subject_name)):
        subject_images_dir = os.path.join(unmask_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            img_path = os.path.join(subject_images_dir, img_name)
            img = face_recognition.load_image_file(img_path)
            temp_x_list.append(img)
        X_total.append(temp_x_list)

X_masked_features = []
for x_list in tqdm.tqdm(X_masked, desc="HOG masked images", unit="image"):
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False)
        temp_X_features.append(x_feature)
    X_masked_features.append(temp_X_features)

X_processed_features = []

for x_list in tqdm.tqdm(X_processed, desc="HOG processed images", unit="image"):
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False)
        temp_X_features.append(x_feature)
    X_processed_features.append(temp_X_features)

# Flatten the lists
X_masked_flat = [feature for sublist in X_masked_features for feature in sublist]
X_processed_flat = [feature for sublist in X_processed_features for feature in sublist]
# print(len(X_masked_flat), len(X_processed_flat))

# X_total = X_masked + X_processed
X_flat = [feature for sublist in X_total for feature in sublist]

# Creating labels for masked and processed images
y_masked = [1] * len(X_masked_flat)
y_processed = [0] * len(X_processed_flat)
# print(len(y_masked), len(y_processed))

# Combining masked and processed features and labels
X_if_mask = X_masked_flat + X_processed_flat
y_if_mask = y_masked + y_processed
# print(len(X_if_wear_mask), len(y_if_wear_mask))

y_test_if_employee = [1] * 450 + [0] * 300 + [1] * 450 + [0] * 300

# print(len(X_if_mask), len(y_if_mask), len(y_test_if_employee), len(X_flat))
# Split the dataset into training and testing sets (80-20 split)
# X_train, X_test, y_train_if_mask, y_test_if_mask, y_test_if_employee_train, y_test_if_employee_test = split_and_shuffle_lists(X_if_mask, y_if_mask, y_test_if_employee, test_size=0.2, random_state=42)
X_train, X_test, y_train_if_mask, y_test_if_mask, y_test_if_employee_train, y_test_if_employee_test, X_ori_train, X_ori_test = split_and_shuffle_lists(X_if_mask, y_if_mask, y_test_if_employee, X_flat, test_size=0.2, random_state=42)


# Train the SVM classifier
if_mask_classifier = SVC(kernel='linear')

# print(type(X_train), type(y_train_if_mask))
# print(X_train, y_train_if_mask)
# print(len(y_if_mask))
# print(len(y_test_if_mask), len(y_train_if_mask))

print("Training the SVM classifier...") 
if_mask_classifier.fit(X_train, y_train_if_mask)

print("Testing the classifier...")
y_pred_if_mask = if_mask_classifier.predict(X_test)

# Evaluate the classifier
wear_mask_accuracy = accuracy_score(y_test_if_mask, y_pred_if_mask)
# print(f"Accuracy: {wear_mask_accuracy * 100:.2f}%")

X_mask = []
X_unmask = []
y_mask = []
y_unmask = []

for i, pred in enumerate(y_pred_if_mask):
    if pred == 1:
        X_mask.append(X_ori_test[i])
        y_mask.append(y_test_if_employee_test[i])
    else:
        X_unmask.append(X_ori_test[i])
        y_unmask.append(y_test_if_employee_test[i])

unmask_processedset_path = "GeorgiaTechFaces/gray_1"
X_processed = []
y = []

for subject_name in tqdm.tqdm(os.listdir(unmask_processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(processedset_path, subject_name)):
        subject_images_dir = os.path.join(processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg') and 1 <= int(img_name.split('.')[0]) <= 10:
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X_processed.append(temp_x_list)
        

X_employee = X_processed[0:30]
X_outsider = X_processed[30:]

# Train a face recognizer on the Employee set
unmask_employee_encodings = []

# for employee_images in X_employee:
for employee_images in tqdm.tqdm(X_employee, desc='employee training without mask'):
    for image in employee_images:
        face_locations = face_recognition.face_locations(image)
        employee_encoding = face_recognition.face_encodings(image, face_locations)
        if employee_encoding != []:
            unmask_employee_encodings.append(employee_encoding)
        # use one picture to train for one person
        # break
    
    
    
mask_processedset_path = "GeorgiaTechFaces/Maskedgray_1"

X_processed = []
y = []

for subject_name in tqdm.tqdm(os.listdir(mask_processedset_path), desc='reading processed images'):
    if os.path.isdir(os.path.join(mask_processedset_path, subject_name)):
        subject_images_dir = os.path.join(mask_processedset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg') and 1 <= int(img_name.split('.')[0]) <= 10:
                img_path = os.path.join(subject_images_dir, img_name)
                img = face_recognition.load_image_file(img_path)
                temp_x_list.append(img)
        X_processed.append(temp_x_list)
        
X_employee = X_processed[0:30]
# Train a face recognizer on the Employee set
mask_employee_encodings = []

# for employee_images in X_employee:
for employee_images in tqdm.tqdm(X_employee, desc='employee training with mask'):
    for image in employee_images:
        face_locations = face_recognition.face_locations(image)
        employee_encoding = face_recognition.face_encodings(image, face_locations)
        if employee_encoding != []:
            mask_employee_encodings.append(employee_encoding)
        # use one picture to train for one person
        # break

print(f"if_wear_mask_accuracy: {(wear_mask_accuracy*100):.2f}%")

y_probe_mask = []
y_probe_unmask = []

order = 0
# test in unmask images
for image in X_unmask:
    print(f"Processing image {order}")
    order += 1
    flag = 0
    probe_encoding = face_recognition.face_encodings(image)
    for encoding in unmask_employee_encodings:
        if probe_encoding == []:
            break
        encoding = np.array(encoding)
        # change the tolerance value here to get the best result
        results = face_recognition.compare_faces(encoding, probe_encoding, tolerance=0.35)
        if len(results) and results[0]:
            if flag == 0:
                print("ACCEPT")
                y_probe_unmask.append(1)
                flag = 1
        else:
            pass
    if flag:
        continue
    print("REJECTED")
    y_probe_unmask.append(0)

# print("y_probe", y_probe)
# print("y_test", y_test)

# Evaluate the classifier
unmask_accuracy = accuracy_score(y_unmask, y_probe_unmask)

order = 0
# test in unmask images
for image in X_mask:
    print(f"Processing image {order}")
    order += 1
    flag = 0
    probe_encoding = face_recognition.face_encodings(image)
    for encoding in mask_employee_encodings:
        if probe_encoding == []:
            break
        encoding = np.array(encoding)
        # change the tolerance value here to get the best result
        results = face_recognition.compare_faces(encoding, probe_encoding, tolerance=0.5)
        if len(results) and results[0]:
            if flag == 0:
                print("ACCEPT")
                y_probe_mask.append(1)
                flag = 1
        else:
            pass
    if flag:
        continue
    print("REJECTED")
    y_probe_mask.append(0)

# print("y_probe", y_probe)
# print("y_test", y_test)

# Evaluate the classifier
mask_accuracy = accuracy_score(y_mask, y_probe_mask)
print(f"if_wear_mask_accuracy: {(wear_mask_accuracy*100):.2f}%")
print(f"unmask_accuracy: {unmask_accuracy * 100:.2f}%")
print(f"mask_accuracy: {mask_accuracy * 100:.2f}%")
plot_confusion_matrix(y_mask, y_probe_mask, ['Employee', 'Outsider'])

# if_wear_mask_accuracy: 100.00%
# unmask_accuracy: 96.73%
# mask_accuracy: 62.59% (0.4)
