#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import mobilenet_v2
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2


# In[2]:


path_actor = '/home/user/Naman/Facescrub/actors/faces/'


# In[17]:


x = np.zeros([15385,100,100,3])
y = []
#x_test = np.zeros([1026,100,100,3])
#y_test = []


# In[19]:


transform = transforms.Compose([transforms.ToTensor()])
i=0
for folders in os.listdir(path_actor):
    for im in os.listdir(path_actor+"/"+folders):
        img = image.load_img(path_actor+folders+"/"+im, target_size=(100, 100,3))
        x_ = transform(img)
        x_ = x_.permute(1,2,0)
        x[i] = x_
        y.append(folders)
        i += 1        


# In[20]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
le = preprocessing.LabelEncoder()
y_fin = le.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_fin = y_fin.reshape(len(y_fin), 1)
onehot_encoded = onehot_encoder.fit_transform(y_fin)
#y_tst = le.fit_transform(y_test)
#y_tst = y_tst.reshape(len(y_tst), 1)
#onehot_encoded_test = onehot_encoder.fit_transform(y_tst)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,onehot_encoded, test_size=0.15, random_state=42)


# In[24]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense

# Set up GPU device
device_name = '/device:GPU:1' # Point to the desired GPU device
if not tf.config.list_physical_devices('GPU'):
    print("No GPU devices found. Using CPU instead.")
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices, device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Found GPU at: {}'.format(device_name))

# Load ResNet50 model without the top layer
resnet101_model = ResNet101(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Add a new output layer with 210 neurons
x = resnet101_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(210, activation='softmax')(x)
model = tf.keras.Model(inputs=resnet101_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Run model on the desired GPU device with input as x_train
with tf.device(device_name):
    model.fit(x_train, y_train, batch_size=64, epochs=75, validation_split=0.2)

model.save('/home/user/Naman/model_resnet_face_101_2')

y_out = model.predict(x_test)
count=0
for i in range(2308):
    if np.argmax(y_out[i]) == np.argmax(y_test[i]):
        count += 1
print("Correct predicted images : ",count)


