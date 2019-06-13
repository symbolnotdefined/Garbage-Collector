#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:45:52 2018

@author: aman
"""

import keras
from keras.models import load_model
from keras.models import Sequential
import cv2
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = Sequential()

model =load_model('first_try.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('paper.jpeg')
#img = cv2.resize(img,(0,0), fx=0.5, fy=0.5)
#img = np.reshape(img,[1,150,150,3])
classes = model.predict_classes(img)[0]
print (classes)

#img = np.reshape(img,[1,150,150,3])
#classes = model.predict_classes(img,img1)
#print(classes)