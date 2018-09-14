#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:07:23 2018

@author: prayash
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


test_image = Image.open('/Users/prayash/Downloads/MitchellJohnson.jpg')
img_test = test_image.resize((img_rows,img_cols))
gray = img_test.convert('L')
img_test_array = array(gray)
m_test,n_test = img_test_array.shape[0:2]
plt.imshow(img_test_array)

immatrix_test = array(img_test_array.flatten(),'f')
immatrix_test.shape


img_test_bowler = immatrix_test.reshape(1, 1, img_rows, img_cols)
img_test_bowler = img_test_bowler.astype('float32')

y_pred_test = model.predict_classes(img_test_bowler)
print("And the player in the picture seems to be a :")
if y_pred_test == [0]:
    print("Legspinner")
elif y_pred_test == [1]:
    print("FastBowler")
elif y_pred_test == [2]:
    print("OffSpinner")
elif y_pred_test == [3]:
    print("SwingBowler")
elif y_pred_test == [4]:
    print("Slow-Arm-Chinamen")