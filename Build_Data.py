# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:11:33 2019

@author: Hanan Bahy1
"""

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

                                       #1-#########

path_training = 'AUGMENTED'
seborrheic_melanoma_datagen = ImageDataGenerator(horizontal_flip=True,     #each image will repeated 6 times
                                      vertical_flip=True,
                                      zoom_range=.1,
                                      shear_range=0.2,
                                      rotation_range=45)
                                    

nevus_datagen = ImageDataGenerator()#(horizontal_flip=True)      #each image will repeated 1times

   

                        #2-   ###################
melanoma_class = seborrheic_melanoma_datagen.flow_from_directory(path_training,
                                                 target_size = (224,224),
                                                 classes=['melanoma'],
                                                 batch_size=374,
                                                 save_to_dir='AUGMENTED/melanoma')   #the size of input data to our CNN

seborrheic_keratosis_class = seborrheic_malenoma_datagen.flow_from_directory(path_training,
                                                 target_size = (224,224),
                                                 classes=['seborrheic_keratosis'],
                                                 batch_size=254,
                                                 save_to_dir='AUGMENTED/seborrheic_keratosis')   #the size of input data to our CNN

nevus_class = nevus_datagen.flow_from_directory(path_training,
                                                 target_size = (224,224),
                                                 classes =['nevus'],
                                                 batch_size=1372,
                                                 save_to_dir='AUGMENTED/nevus')

# 3-######                                               
for i in range(6):          
    seborrheic_keratosis_class.next()
    melanoma_class.next()
    if i<1:
        nevus_class.next()       
#        if i <1:
#            malenoma_class.next()

####from Test data
            
        #1-
path='test'
test_nevus_datagen = ImageDataGenerator(rotation_range=45,
                                  horizontal_flip=True,     #each image will repeated 5times
                                      vertical_flip=True,
                                      shear_range=0.2)#,
                                      #width_shift_range=0.5, height_shift_range=0.5)#,

test_seborrheic_melanoma_datagen = ImageDataGenerator(rotation_range=45,
                                  horizontal_flip=True,     #each image will repeated 6times
                                  vertical_flip=True,
                                  zoom_range=.1,
                                  shear_range=0.2)#,


                                 
        ###2-
seborrheic_keratosis_class =  test_seborrheic_melanoma_datagen.flow_from_directory(path,
                                                 target_size = (224,224),
                                                 classes=['seborrheic_keratosis'],
                                                 batch_size=90,
                                                 save_to_dir='AUGMENTED/seborrheic_keratosis')   #the size of input data to our CNN

nevus_class = test_nevus_datagen.flow_from_directory(path,
                                                 target_size = (224,224),
                                                 classes =['nevus'],
                                                 batch_size=393,
                                                 save_to_dir='AUGMENTED/nevus')

melanoma_class = test_seborrheic_melanoma_datagen.flow_from_directory(path,
                                                 target_size = (224,224),
                                                 classes=['melanoma'],
                                                 batch_size=117,
                                                 save_to_dir='AUGMENTED/melanoma')   #the size of input data to our CNN

     ###3-
for i in range(6):          
    seborrheic_keratosis_class.next()
    melanoma_class.next()
    if i <5:
            nevus_class.next()
    
    