import tensorflow as tf
#import tf.keras.utils
plot_model = tf.keras.utils.plot_model
import pickle
import sys
import numpy as np
import pandas as pd
from cv2 import imread
import os
import matplotlib.pyplot as plt

import cv2
import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import RMSprop

from sklearn.utils import shuffle

import numpy.random as rng

import face_recognition
#from google.colab.patches import cv2_imshow
import os
import cv2

train_folder = '/content/SlowFastData/demo/AVA/face_recog'
test_folder = '/content/SlowFast/slowfast/visualization/'

images = []
names = []
for person in os.listdir(train_folder) :
    person_path = os.path.join(train_folder,person)

    for filename in os.listdir(person_path) :
        image_path = os.path.join(person_path, filename)
        curr = cv2.imread(image_path)
        images.append(curr)
        names.append(person) # + '_' + filename)
print(names)

def find_encodings(images) :
    encoded_list = []
    count = 1
    for img, nm in zip(images, names) :
        print(nm)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        #plt.show()
        #cv2_imshow(img)
        
        encodings = face_recognition.face_encodings(img)
        if encodings :
            print('encoding created')
            encode = face_recognition.face_encodings(img)[0]
        else : continue
        encoded_list.append(encode)
        count += 1
    print('***************', len(encoded_list))
    return encoded_list

encoding_known = find_encodings(images)

files = [f for f in os.listdir(test_folder) if '.png' in f or '.jpg' in f]

for fname in files :
    print('\n\nfilename--->', fname, '....................')
    img = cv2.imread(test_folder + fname)

    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.show()

    faces_current = face_recognition.face_locations(img)
    print('No of faces ---> ', len(faces_current))
    encode_current = face_recognition.face_encodings(img, faces_current)

    boundaries = []
    person_names = []
    for encodeF, faceLoc in zip(encode_current, faces_current):
        #print('---------------------------------')
        matches = face_recognition.compare_faces(encoding_known, encodeF, tolerance=0.5)
        #print(matches)
        #input()
        face_dis = face_recognition.face_distance(encoding_known, encodeF)
        #print(face_dis)
        #input()
        matched_index = np.argmin(face_dis)
        #print(matched_index)
        #print(matches[matched_index])

        if matches[matched_index]:
        	name = names[matched_index].upper()
        else :
            name = 'unknown'

        print('names is --->', name)
        top, right, bottom, left = faceLoc
        bndry = left, top, right, bottom
        boundaries.append(bndry)
        person_names.append(name)

        #face = img[top:bottom, left:right, :]
        #plt.imshow(face)
        #plt.show()
        #cv2.imshow('f', face)
        #cv2.waitKey(3000)

    for bndry, person in zip(boundaries, person_names) :
        cv2.rectangle(img, (bndry[0], bndry[1]), (bndry[2], bndry[3]), (0, 255, 0), 5)
        cv2.putText(img, person, (bndry[0], bndry[1] - 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    #plt.imshow(img)
    #plt.show()
