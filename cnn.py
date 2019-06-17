#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:19:31 2019

@author: nishtha
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier=Sequential()

#STEP1:Convolution(32 filters of 3*3 and output images are of 64*64 pixels and 3 boxes coz its colored and if they were black and white then 1 box)
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step2:pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step add more layers(2-3 aadd krlo)
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3: flatten
classifier.add(Flatten())

#step4:full connection
classifier.add(Dense(output_dim=128,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#from keras documentation:imgae preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64), #agar 64 se bada number lemge toh more accuracy
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



#PREDICTIONSSSS
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'


