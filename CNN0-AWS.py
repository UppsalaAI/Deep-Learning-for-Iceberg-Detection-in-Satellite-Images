#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import numpy as np
import pandas as pd
import pylab
import sagemaker
import keras
from sagemaker import get_execution_role
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,ReduceLROnPlateau


# In[2]:


role = get_execution_role()
bucket='sagemaker-eu-north-1-845746890132'
data_key1='train.json'
data_location1='s3://{}/{}'.format(bucket,data_key1)
pd.read_json(data_location1)


# In[3]:


#Load data
plt.rcParams['figure.figsize'] = 10, 10
train = pd.read_json(data_location1)
train.inc_angle = train.inc_angle.replace('na',0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
print('done!')


# In[4]:


#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
X_angle_train=np.array(train.inc_angle)
#y_train=np.array(train["is_iceberg"]
X_train.shape

#define the model
def getModel():
    gmodel=Sequential()
    
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    
    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    
    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    
    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))
    
    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())
    
    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    
    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))
    
    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
   # mypotim=Adam(lr=0.001,decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


# In[5]:


#get ready to train the model    
def get_callbacks(filepath, patience=3):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',factor= 0.1,patience=7,verbose=1,epsilon=1e-4, mode ='min')
    return [es, msave,reduce_lr_loss]

file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

   
y_train=np.array(train['is_iceberg'])
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1, train_size=0.75) 
    
    
#Without denoising, core features
import os
gmodel=getModel()
gmodel.fit(X_train_cv, y_train_cv,
          batch_size=32,
          epochs=50,
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

gmodel.load_weights(file_path)
score = gmodel.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




