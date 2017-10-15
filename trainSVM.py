import os
from skimage import data, color, exposure
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16, 4))

PERSON_WIDTH = 64
PERSON_HEIGHT = 128
leftop = [16, 16]
rightbottom = [16 + PERSON_WIDTH, 16 + PERSON_HEIGHT]

pos_img_dir = 'C:/Users/szymo/Desktop/INRIAPerson/train_64x128_H96/pos/'
neg_img_dir = 'C:/Users/szymo/Desktop/INRIAPerson/train_64x128_H96/neg/'
pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)

X = []
y = []
print('start loading ' + str(len(pos_img_files)) + ' positive files')
for pos_img_file in pos_img_files:
    pos_filepath = pos_img_dir + pos_img_file
    pos_img = data.imread(pos_filepath, as_grey=True)
    pos_roi = pos_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(pos_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    X.append(fd)
    y.append(1)
print('start loading ' + str(len(neg_img_files)) + ' negative files')
for neg_img_file in neg_img_files:
    neg_filepath = neg_img_dir + neg_img_file
    neg_img = data.imread(neg_filepath, as_grey=True)
    neg_roi = neg_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(neg_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    X.append(fd)
    y.append(0)

## covert list into numpy array
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

from sklearn import svm
from sklearn.externals import joblib

print('start learning SVM.')
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
clf = svm.SVC()
clf.fit(X, y)
print('finish learning SVM.')
joblib.dump(lin_clf, 'person_detector.pkl', compress=9)
"""
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.externals import joblib
print('start learning SVM.')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10**i for i in range(-4,0)], 'C': [10**i for i in range(1,4)]}]
gscv = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring="mean_squared_error")
gscv.fit(X_train, y_train)
#一番スコア悪い&良い奴を出す
params_min,_,_ = gscv.grid_scores_[np.argmin([x[1] for x in gscv.grid_scores_])]
svm_best = gscv.best_estimator_
print('start re-learning SVM with best parameter set.')
svm_best.fit(X_train, y_train)

print('finish learning SVM　with Grid-search and cross-varidation.')

print('searched result of  C =')

print('searched result of  gamma =')
"""