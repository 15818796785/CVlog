import os

from sklearn.naive_bayes import GaussianNB

from CVlog.bonus.utils import load_image_dataset, read_200000_Y, shuffle, save_model, load_model, test_model, \
    train_model_in_batch

dataset_path = "../img_align_celeba"
model_path = "../model_ckpts/200000_gaussian_nb_model.pkl"
filename = '../identity_CelebA.txt'
X_train = []
X_train = load_image_dataset(dataset_path)
Y_train = read_200000_Y(filename)

shuffle(X_train, Y_train)

# 创建朴素贝叶斯分类器
current_epoch = 0
if not os.path.exists(model_path):
    model = GaussianNB()
    train_model_in_batch(model,X_train, Y_train, current_epoch)
    save_model(model, model_path)
elif (current_epoch<5):
    model = load_model(model_path)
    train_model_in_batch(model, X_train, Y_train, current_epoch)
    save_model(model, model_path)
else:
    model = load_model(model_path)


# saved_path = "../model_ckpts/"
# test_model(model, X_test, Y_test,saved_path,"Bayes_200000_Datasets.txt")


