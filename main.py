# Import all modules
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
from sklearn import cross_validation
from sklearn import metrics
import imutils


rootDir = "C:/Users/szymo/Desktop/INRIAPerson"
trainDir = "C:/Users/szymo/Desktop/INRIAPerson/Train"
testDir = "C:/Users/szymon/Desktop/INRIAPerson/Test"

win_size = (64,128)

train_X=[]
train_Y=[]
test_complex_X=[]
test_complex_Y=[]
test_uniform_X=[]
test_uniform_Y=[]

labels = os.listdir(trainDir)


# Initialize the HOG descriptor/person detector

desc = cv2.HOGDescriptor()


############################################################
############### Training Data ##############################
############################################################

# Get features and labels of training data

for i
