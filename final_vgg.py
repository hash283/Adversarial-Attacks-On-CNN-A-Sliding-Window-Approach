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
break
 transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
        img_tensor = transform(img)
        x_ = image.img_to_array(img)
        x_ = preprocess_input(x_)
        x.append(x_,axis=0)
        torch.cat((x, x_), dim=1)
        x[i] = x_
        i += 1
        x = np.concatenate((x, x_), axis = 0)
        y.append(folders)
        if (person_images_idx == X_rand[rand_idx]):
                X_test[ts_idx] = x
                y_ts[ts_idx] = personFolder
                ts_idx += 1
                #print(ts_idx)
                if rand_idx<number_of_images-1:
                    rand_idx += 1
                
            else:
                X_train[tr_idx] = x
                #print("else\n")
                y_tr[tr_idx] = personFolder
                tr_idx += 1
            count += 1
            person_images_idx += 1
            #print(count)
            if (count % ((number_of_images*aug_multiplier)) == 0):
                print("Processing image: ", count, ", ", img)
                break
print(k)


# In[20]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
le = preprocessing.LabelEncoder()
y_fin = le.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_fin = y_fin.reshape(len(y_fin), 1)
onehot_encoded = onehot_encoder.fit_transform(y_fin)
y_tst = le.fit_transform(y_test)
y_tst = y_tst.reshape(len(y_tst), 1)
onehot_encoded_test = onehot_encoder.fit_transform(y_tst)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,onehot_encoded, test_size=0.15, random_state=42)


# In[24]:


import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
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
#vgg16_model = VGG19(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Add a new output layer with 210 neurons
#xy = vgg16_model.output
#xy = tf.keras.layers.GlobalAveragePooling2D()(xy)
#xy = Dense(210, activation='softmax')(xy)
#model = tf.keras.Model(inputs=vgg16_model.input, outputs=xy)


model = mobilenet_v2.MobileNetV2(input_shape=(100, 100,3), weights=None, include_top=True, alpha=1., classes=210)

#model = mobilenet_v2(weights='imagenet', include_top=True, input_shape=(100, 100, 3),num_classes=210)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Run model on the desired GPU device with input as x_train
with tf.device(device_name):
    model.fit(x_train, y_train, batch_size=256, epochs=800, validation_split=0.20)

model.save('/home/user/Naman/model_vgg16_face_2')

y_out = model.predict(x_test)
count=0
for i in range(2308):
    if np.argmax(y_out[i]) == np.argmax(y_test[i]):
        count += 1
print("Correct predicted images : ",count)

