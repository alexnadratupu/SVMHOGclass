import os
pos_img_dir = 'C:/Users/szymo/Desktop/INRIAPerson/train_64x128_H96/pos/'
neg_img_dir = 'C:/Users/szymo/Desktop/INRIAPerson/train_64x128_H96/neg/'

pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)

from skimage import data
filepath = pos_img_dir + pos_img_files[0]
img = data.imread(filepath,as_grey=True)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#For embedding plotted figures on the notebook

plt.figure(figsize=(8, 4))
plt.subplot(121).set_axis_off()
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Input image')

PERSON_WIDTH = 64
PERSON_HEIGHT = 128
leftop = [16,16]
rightbottom =  [16+PERSON_WIDTH,16+PERSON_HEIGHT]

roi = img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
plt.subplot(122).set_axis_off()
plt.imshow(roi, cmap=plt.cm.gray)
plt.title('ROI region')
plt.show()