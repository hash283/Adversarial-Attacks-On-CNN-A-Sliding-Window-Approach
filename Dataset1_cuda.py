#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[13]:


path_actor = '/home/user/Naman/Facescrub/actors/images/'
#/data/home/kpsingh/Naman


# In[14]:


#x = torch.tensor()
x = np.zeros([15343,200,200,3])
y = []
i = 0


# In[15]:


transform = transforms.Compose([transforms.ToTensor()])


# In[16]:


#transform = transforms.ToTensor()

for folders in os.listdir(path_actor):
    for im in os.listdir(path_actor+"/"+folders):
        img = image.load_img(path_actor+folders+"/"+im, target_size=(200, 200,3))
        x_ = transform(img)
        x_ = x_.permute(1,2,0)
        x[i] = x_.numpy()
        i += 1
        #x = torch.cat([x, x_], dim=0)
        y.append(folders)
        #transform = transforms.Compose([transforms.PILToTensor()])
        #break
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
        #img_tensor = transform(img)
        #x_ = image.img_to_array(img)
        #x_ = preprocess_input(x_)
        #x.append(x_,axis=0)
        #torch.cat((x, x_), dim=1)
        #x[i] = x_
        #i += 1
        #x = np.concatenate((x, x_), axis = 0)
        #y.append(folders)
#         if (person_images_idx == X_rand[rand_idx]):
#                 X_test[ts_idx] = x
#                 y_ts[ts_idx] = personFolder
#                 ts_idx += 1
#                 #print(ts_idx)
#                 if rand_idx<number_of_images-1:
#                     rand_idx += 1
                
#             else:
#                 X_train[tr_idx] = x
#                 #print("else\n")
#                 y_tr[tr_idx] = personFolder
#                 tr_idx += 1
#             count += 1
#             person_images_idx += 1
#             #print(count)
#             if (count % ((number_of_images*aug_multiplier)) == 0):
#                 print("Processing image: ", count, ", ", img)
#                 break


# In[17]:


import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
y_fin = le.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_fin = y_fin.reshape(len(y_fin), 1)
onehot_encoded = onehot_encoder.fit_transform(y_fin)


# In[18]:


x[0].shape


# In[47]:


num_classes = len(set(y))
num_classes
nr_classes = num_classes


# In[48]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator()


# In[50]:


# Define a generator function that yields batches of NumPy arrays
def batch_generator(x, y, batch_size):
    num_batches = len(x) // batch_size
    while True:
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            yield x[start_index:end_index], y[start_index:end_index]

# Create a TensorFlow Dataset object using the generator function
dataset = tf.data.Dataset.from_generator(
    generator=lambda: batch_generator(x, onehot_encoded, batch_size=64),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None,200, 200, 3], [None, num_classes])
)

# Enable GPU acceleration

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
#        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 9)]
        )
    except RuntimeError as e:
        print(e)

#gpus = tf.config.experimental.list_physical_devices('GPU')


# Train the model on the dataset
#model1 = mobilenet_v2.MobileNetV2(input_shape=(200, 200,3), weights=None, include_top=True, alpha=1., classes=nr_classes)
#model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
#               metrics=[categorical_crossentropy, categorical_accuracy])
#model1.fit(dataset,steps_per_epoch=len(x) // 32,epochs=300, verbose=1)
#model1.save('/home/user/Naman/model_1')


# In[45]:


model2 = mobilenet_v2.MobileNetV2(input_shape=(200, 200,3), weights=None, include_top=True, alpha=1., classes=nr_classes)
#if torch.cuda.is_available():
#    model.cuda()
model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
               metrics=[categorical_crossentropy, categorical_accuracy])
#model.load_model("../working/model_25gpunotop.h5")


# In[46]:


model2.fit(dataset,steps_per_epoch=len(x) // 32,epochs=400, verbose=1)
model2.save('/home/user/Naman/model_2')


# In[ ]:


model3 = mobilenet.MobileNet(input_shape=(200, 200,3), weights=None, include_top=True, alpha=1., classes=nr_classes)
#if torch.cuda.is_available():
#    model.cuda()
model3.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
               metrics=[categorical_crossentropy, categorical_accuracy])


# In[ ]:


model3.fit(dataset,steps_per_epoch=len(x) // 32,epochs=300, verbose=1)
model3.save('/home/user/Naman/model_3')


# In[ ]:


model4 = mobilenet.MobileNet(input_shape=(200, 200,3), weights=None, include_top=True, alpha=1., classes=nr_classes)
#if torch.cuda.is_available():
#    model.cuda()
model4.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
               metrics=[categorical_crossentropy, categorical_accuracy])


# In[ ]:


model4.fit(dataset,steps_per_epoch=len(x) // 32,epochs=400, verbose=1)
model4.save('/home/user/Naman/model_4')

