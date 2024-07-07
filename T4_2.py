from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import tqdm
import os
import cv2
import numpy as np

dataset_path = "GeorgiaTechFaces/Maskedset_1"

X_masked = []
y = []
for subject_name in tqdm.tqdm(os.listdir(dataset_path), desc='reading images'):
    if os.path.isdir(os.path.join(dataset_path, subject_name)):
        y.append(subject_name)
        subject_images_dir = os.path.join(dataset_path, subject_name)
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(subject_images_dir, img_name)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_x_list.append(gray_img)
        X_masked.append(temp_x_list)

X_features = []
for x_list in X_masked:
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False)
        temp_X_features.append(x_feature)
    X_features.append(temp_X_features)

# Flattening the X_features list
X_processed = []
for features in X_features:
    X_processed.extend(features)

# Assuming y contains labels indicating if a mask is present or not
# For demonstration purposes, let's assume y contains binary labels 0 (no mask) or 1 (mask)
# Adjust the labels and their extraction logic based on the actual dataset

# Flattening y to match the length of X_processed
y_flattened = [label for label, features in zip(y, X_features) for _ in features]

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_flattened, test_size=0.2, shuffle=True, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Test the classifier
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")