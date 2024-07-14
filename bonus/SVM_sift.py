import os
import tqdm
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.vq import vq

# 数据集路径
dataset_path = 'GeorgiaTechFaces/ConvertGrayscaleprocessedset_1'
masked_path = 'GeorgiaTechFaces/ConvertGrayscaleMaskprocessedset_1'

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.zeros((1, sift.descriptorSize()), dtype=np.float32)
    return keypoints, descriptors

def load_and_extract_sift_features(path, desc):
    descriptors_list = []
    y = []
    keypoints_list = []
    for subject_name in tqdm.tqdm(os.listdir(path), desc=desc):
        if os.path.isdir(os.path.join(path, subject_name)):
            subject_images_dir = os.path.join(path, subject_name)
            for img_name in os.listdir(subject_images_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(subject_images_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (150, 150))
                    keypoints, descriptors = extract_sift_features(img)
                    descriptors_list.append(descriptors)
                    y.append(int(subject_name[1:]))
                    keypoints_list.append((img, keypoints))  # 保存图像和关键点
    return descriptors_list, np.array(y), keypoints_list

# 读取并提取SIFT特征
descriptors_list_train, y_train, keypoints_list_train = load_and_extract_sift_features(dataset_path, 'reading unmasked images')
descriptors_list_test, y_test, keypoints_list_test = load_and_extract_sift_features(masked_path, 'reading masked images')
path = 'sift'
if not os.path.exists(path):
    os.makedirs(path)
# 展示部分图像的SIFT关键点和描述子
def visualize_sift_keypoints(keypoints_list, num_samples=5, folder = 'sift'):
    for i, (img, keypoints) in enumerate(keypoints_list[:num_samples]):
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f'SIFT Keypoints - Sample {i+1}')
        plt.axis('off')
        pathimg = os.path.join(folder, f'SIFT_{i+1}.png')
        plt.savefig(pathimg)
        plt.show()

visualize_sift_keypoints(keypoints_list_train)

# 创建所有图像的描述符堆
all_descriptors = np.vstack(descriptors_list_train + descriptors_list_test)

print('Stack of descriptors done')

# 使用PCA减少特征维度
pca = PCA(n_components=5)  # 可以根据需要调整主成分数量
all_descriptors_pca = pca.fit_transform(all_descriptors)

# 使用MiniBatchKMeans进行聚类
num_clusters = 5000  # 可以根据需要调整聚类数量
minibatch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1000, n_init=10)
minibatch_kmeans.fit(all_descriptors_pca)

def compute_bow_histogram(descriptors, kmeans, pca):
    descriptors_pca = pca.transform(descriptors)
    words, _ = vq(descriptors_pca, kmeans.cluster_centers_)
    histogram, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
    return histogram

# 计算BoW直方图
X_train = np.array([compute_bow_histogram(desc, minibatch_kmeans, pca) for desc in descriptors_list_train])
X_test = np.array([compute_bow_histogram(desc, minibatch_kmeans, pca) for desc in descriptors_list_test])

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义并训练SVM模型
svm = SVC(C=0.1, kernel='linear', gamma='auto')
svm.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = svm.predict(X_test)

# 计算并打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 打印分类报告
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 保存混淆矩阵和分类报告
conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(y_test), columns=np.unique(y_test))
conf_matrix_df.to_csv('confusion_matrix_SVM_sift.csv', index=True)

with open('classification_report_SVM_sift.txt', 'w') as f:
    f.write(class_report)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_SVM_sift.png')
plt.show()
