import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from starter import X_processed, X_masked

# 假设 X_processed 是处理过的正常人脸图像的特征列表
# 假设 X_masked 是带有面具的人脸图像的特征列表
# y 是包含图像标签的列表，'mask' 表示戴面具，'no_mask' 表示不戴面具

# 将正常和带面具的数据合并

X = np.concatenate([X_processed, X_masked])
y = ['no_mask'] * len(X_processed) + ['mask'] * len(X_masked)



# 分割数据为训练集和测试集，比例为80%训练，20%测试
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 初始化 SVM 分类器
classifier = svm.SVC(kernel='linear')  # 使用线性核

# 训练模型
classifier.fit(x_train, y_train)

# 使用测试集预测结果
y_pred = classifier.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"分类器准确率：{accuracy * 100:.2f}%")
