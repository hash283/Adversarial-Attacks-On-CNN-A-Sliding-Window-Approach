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
from torchvision import models
#import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2


# In[2]:


path_actor = '/home/user/Naman/Facescrub/actors/faces/'


# In[3]:


x = np.zeros([15385,100,100,3])
y = []


# In[4]:


transform = transforms.ToTensor()
i=0
transform = transforms.Compose([transforms.ToTensor()])
for folders in os.listdir(path_actor):
    for im in os.listdir(path_actor+"/"+folders):
        img = image.load_img(path_actor+folders+"/"+im, target_size=(100, 100,3))
        x_ = transform(img)
        x_ = x_.permute(1,2,0)
        x[i] = x_
        y.append(folders)
        i+=1
        x = torch.cat([x, x_], dim=0)
        
        transform = transforms.Compose([transforms.PILToTensor()])
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


# In[5]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
le = preprocessing.LabelEncoder()
y_fin = le.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_fin = y_fin.reshape(len(y_fin), 1)
onehot_encoded = onehot_encoder.fit_transform(y_fin)


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,onehot_encoded, test_size=0.15, random_state=42)


# In[7]:


n_model = tf.keras.models.load_model('/home/user/Naman/model_resnet_face_101')


# In[8]:


y_out = n_model.predict(x_test)


# In[9]:


x_test.shape


# In[10]:


count=0
y_curr = []
x_curr = []
for i in range(2308):
    if np.argmax(y_out[i]) == np.argmax(y_test[i]):
        y_curr.append(y_test[i])
        x_curr.append(x_test[i])
        count += 1
print("Correct Output Images : ",count)


# In[11]:


loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        input_image = tf.convert_to_tensor(input_image)
        tape.watch(input_image)
        label = tf.cast(input_label, tf.int32)
        Xtest = tf.expand_dims(input_image, axis=0)
        ynew = n_model(Xtest)
        ynew = tf.squeeze(ynew)
        loss = loss_object(label,ynew)
    
    gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# In[ ]:





# In[14]:




# descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
#                 for eps in epsilons]
epsilons = [0.01]
for eps in epsilons:    
    curr = 0
    for i in range(x_test.shape[0]):
        #if(x_test[i].size == (1,3,200,200)):
        #    x_test[i] = x_test[i].permute(2,1,0)
        #print(x_test[i].shape)
        perturbations = create_adversarial_pattern(x_test[i], y_test[i])
        #print(perturbations)
        #plt.imshow(perturbations)
        #plt.show()
        adv_x = x_test[i] + eps*perturbations
        #print("adv",adv_x.shape)
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        adv_x = tf.expand_dims(adv_x,axis=0)
        prediction = n_model.predict(adv_x)
        MaxPosition=np.argmax(prediction)
        if MaxPosition == np.argmax(y_test[i]):
            curr += 1
    #prediction_label= df_name.iloc[MaxPosition]
    #print(prediction_label) 
    #display_images(adv_x,MaxPosition)
    print("Current epsilon ",eps,", correct images : ",curr)


# In[12]:


eps = 0.02
def sliding_window_3d(Image, data, window_size,stride,eps):
    # Get dimensions of the data
    height, width, depth = data.shape

    # Get dimensions of the sliding window
    dx , dy  = window_size
    print("Original")
    plt.imshow(Image)
    plt.show()
    ttack_imgs = []
    for i in range(0,height-dx,stride):
        for j in range(0,width-dy,stride):
            image = Image.copy()
            image[i:i+dx,j:j+dy] += eps*data[i:i+dx,j:j+dy]
            attack_imgs.append(image)
            
    return attack_imgs


# In[13]:


count = x_test.shape[0]
count


# In[23]:


 wnd = []
 perturbations = create_adversarial_pattern(x_test[0], y_test[0])
     #plt.imshow(perturbations)
     #plt.show()
 window = sliding_window_3d(x_test[0],perturbations,[30,30],10)
     #wn.append(window)
 wndw = tf.convert_to_tensor(window)
     #print(wndw.shape)
     plt.figure(figsize=(1.5,1.5))
     plt.imshow(wndw[5])
     plt.show()
     for z in window:
         plt.figure(figsize=(1.5,1.5))
         plt.imshow(z)
         plt.show()
 print(wndw)
 for i in wndw:
      plt.figure(figsize=(1.5,1.5))
      plt.imshow(i)
      plt.show()
curr = 0
wnd.append(wndw)
window_out = n_model.predict(wndw)
for j in window_out:
        if np.argmax(j) == np.argmax(y_test[i]):
            curr += 1
    out.append(curr)
    print(curr)


# In[14]:


eps1 = 0.01
epsilon = [0]
for eps in epsilon:
out = []
out_wn = []
for i in range(x_test.shape[0]):
    perturbations = create_adversarial_pattern(x_test[i], y_test[i])
    plt.imshow(perturbations)
    plt.show()
    window = sliding_window_3d(x_test[i],perturbations,[30,30],10)
    wn.append(window)
    wndw = tf.convert_to_tensor(window)
    print(wndw.shape)
    plt.figure(figsize=(1.5,1.5))
    #plt.imshow(wndw[5])
    #plt.show()
    for z in window:
        plt.figure(figsize=(1.5,1.5))
        plt.imshow(z)
        plt.show()
    curr = 0
    out_wn.append(window)
    window_out = n_model.predict(wndw)
    for j in window_out:
        if np.argmax(j) == np.argmax(y_test[i]):
            curr += 1
    out.append(curr)
    print(curr)
tot_sum = sum(out)
print("Total window accuracy for epsilon : ",eps)
print((tot_sum/(count))/max(out))
coun = 0
for j in out:
    if j < max(out):
        coun += 1
print("Number of images for which it is incorrectly classified : ",coun)
     for tst in window:
         adv_x = tf.clip_by_value(tst, -1, 1)
         adv_x = tf.expand_dims(adv_x,axis=0)
         prediction = n_model.predict(adv_x)
         MaxPosition=np.argmax(prediction)
         if MaxPosition == np.argmax(y_img[i]):
             curr += 1
     out.append(curr)
     print(curr)


# In[16]:


wnd_curr = []
for i in range(count):
    if out[i] == 0:
        wnd_curr.append(i)
wnd_curr
        


# In[18]:


print("Total window accuracy for epsilon : ",eps)
print((tot_sum/(count))/max(out))
coun = 0
for j in out:
    if j < np.max(out):
        coun += 1
print("Number of images for which it is incorrectly classified : ",coun)


# In[ ]:


tot = 0
for i in range(100):
    perturbation = create_adversarial_pattern(x_test[i],y_test[i])
    window = sliding_window_3d(x_test[i],perturbation,[20,20],10)
    curr = 0
    pred = n_model.predict(tf.convert_to_tensor(window))
    for x in pred:
        if np.argmax(x) == np.argmax(y_test[i]):
            curr += 1
    
     for tst in window:
         adv_x = tf.clip_by_value(tst, -1, 1)
         adv_x = tf.expand_dims(adv_x,axis=0)
         prediction = n_model.predict(adv_x)
         MaxPosition=np.argmax(prediction)
         if MaxPosition == np.argmax(y_test[i]):
             curr += 1
    print("Total number of windows created : ",len(window))
    print("Correct predicted values : ", curr)
    tot = tot + curr
print()


# In[ ]:


curr = 0
for tst in window:
    adv_x = tf.clip_by_value(tst, -1, 1)
    adv_x = tf.expand_dims(adv_x,axis=0)
    prediction = model.predict(adv_x)
    MaxPosition=np.argmax(prediction)
    if MaxPosition == np.argmax(label_t):
        curr += 1


# In[67]:


print("Total number of windows created : ",len(window))
print("Correct predicted values : ", curr)


# In[ ]:




