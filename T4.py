from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import tqdm
import os
import cv2
import numpy as np

maskedset_path = "GeorgiaTechFaces/Maskedset_1"
processedset_path = "GeorgiaTechFaces/Processedset_1"

X_masked = []
X_processed = []
y = []
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

X_masked_features = []
for x_list in tqdm.tqdm(X_masked, desc="HOG masked images", unit="image"):
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False)
        temp_X_features.append(x_feature)
    X_masked_features.append(temp_X_features)

X_processed_features = []
# for x_list in X_processed:
#     temp_X_features = []
#     for x in x_list:
#         x_feature = hog(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), orientations=8, pixels_per_cell=(10, 10),
#                         cells_per_block=(1, 1), visualize=False)
#         temp_X_features.append(x_feature)
#     X_processed_features.append(temp_X_features)

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

# Creating labels for masked and processed images
y_masked = [1] * len(X_masked_flat)
y_processed = [0] * len(X_processed_flat)

# Combining masked and processed features and labels
X = X_masked_flat + X_processed_flat
y = y_masked + y_processed

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42) #, random_state=42
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')

print("Training the SVM classifier...")
svm_classifier.fit(X_train, y_train)

print("Testing the classifier...")
y_pred = svm_classifier.predict(X_test)
# print(y_test)
# print(y_pred)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")