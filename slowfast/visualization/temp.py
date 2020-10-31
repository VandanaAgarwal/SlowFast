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
import glob

train_folder = '/content/SlowFastData/demo/AVA/face_recog'
test_folder = '/content/SlowFastData/demo/OUTPUT/'

dfs = []
for f in glob.glob(test_folder + '*.csv') :
  df = pd.read_csv(f)
  dfs.append(df)

df_final = pd.concat(dfs, axis=0)
df_final = df_final.sort_values('Task_id')
df_final['Child_name'] = ''

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

def find_encodings(images, names) :
    encoded_list = []
    final_names = []
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
        final_names.append(nm)
        
        count += 1
    print('***************', len(encoded_list))
    return encoded_list, final_names

encoding_known, final_names = find_encodings(images, names)

files = [f for f in os.listdir(test_folder) if '.png' in f or '.jpg' in f]
for fname in files :
    print('\n\nfilename--->', fname, '....................')
    fname = test_folder + fname
    img = cv2.imread(fname)

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
        face_dis = face_recognition.face_distance(encoding_known, encodeF)
        matched_index = np.argmin(face_dis)
        
        if matches[matched_index]:
        	name = final_names[matched_index].upper()
        else :
            name = 'unknown'

        print('names is --->', name)

        df_final.at[df_final.index[df_final['Frame_file']==fname].tolist()[0], 'Child_name'] = name
        
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

df_final.to_csv(test_folder + 'frames_final.csv')
