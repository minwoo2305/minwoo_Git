import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19, ResNet50, NASNetLarge, Xception, InceptionV3, inception_resnet_v2
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import backend as K
import keras
import pandas as pd
from keras.layers.convolutional import Conv2D, MaxPooling2D

flag = 0

while True:
    if flag == 0:
        for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:

            base_dir = 'D:/Data/' + str(num) + '/img_data/2_class/SCI_vs_MCI&AD'

            train_dir = os.path.join(base_dir, 'Train')
            val_dir = os.path.join(base_dir, 'Validation')
            test_dir = os.path.join(base_dir, 'Test')

            train_datagen = ImageDataGenerator(rescale=1. / 255)
            val_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=50,
                                                                class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_dir, target_size=(100, 100), batch_size=50,
                                                            class_mode='binary')
            test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100), batch_size=50,
                                                              class_mode='binary')

            conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

            conv_base.trainable = True

            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

            model = models.Sequential()
            model.add(conv_base)
            model.add(layers.Flatten(name='flatten'))
            model.add(layers.Dense(10, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='auto')
            history = model.fit_generator(train_generator, steps_per_epoch=5, epochs=200,
                                          validation_data=val_generator, validation_steps=10,
                                          callbacks=[early_stopping])

            scores = model.evaluate_generator(test_generator, steps=10)
            acc = round(scores[1] * 100, 1)

            save_path = 'D:/Data/save_model/2_class/SCI_vs_MCI&AD/' + str(num)
            model.save(save_path + '/save_model_' + str(acc) + '.h5')

            K.clear_session()

            flag = 1

    elif flag == 1:
        for num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:

            base_dir = 'D:/Data/' + str(num) + '/img_data/2_class/MCI_vs_AD'

            train_dir = os.path.join(base_dir, 'Train')
            val_dir = os.path.join(base_dir, 'Validation')
            test_dir = os.path.join(base_dir, 'Test')

            train_datagen = ImageDataGenerator(rescale=1. / 255)
            val_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=50,
                                                                class_mode='binary')
            val_generator = val_datagen.flow_from_directory(val_dir, target_size=(100, 100), batch_size=50,
                                                            class_mode='binary')
            test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100), batch_size=50,
                                                              class_mode='binary')

            conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

            conv_base.trainable = True

            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

            model = models.Sequential()
            model.add(conv_base)
            model.add(layers.Flatten(name='flatten'))
            model.add(layers.Dense(10, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='auto')
            history = model.fit_generator(train_generator, steps_per_epoch=10, epochs=200,
                                          validation_data=val_generator, validation_steps=10,
                                          callbacks=[early_stopping])

            scores = model.evaluate_generator(test_generator, steps=10)
            acc = round(scores[1] * 100, 1)

            save_path = 'D:/Data/save_model/2_class/MCI_vs_AD/' + str(num)
            model.save(save_path + '/save_model_' + str(acc) + '.h5')

            K.clear_session()

            flag = 0

