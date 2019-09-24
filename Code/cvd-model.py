import warnings
warnings.filterwarnings("ignore")

from Code.ResidualAttentionNetwork import ResidualAttentionNetwork

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

import h5py

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras import optimizers

from keras.models import load_model

IMAGE_WIDTH=32
IMAGE_HEIGHT=32
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=1
IMAGE_SHAPE=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

batch_size=32

epochs = 500

num_classes = 2

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=.33
)

train_generator = train_datagen.flow_from_directory(
    directory="./dogs-vs-cats/train/", 
    shuffle=True,
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training'
)

valid_generator = train_datagen.flow_from_directory(
    directory="./dogs-vs-cats/train/", 
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode="categorical",
    color_mode='grayscale',
    shuffle=True,
    subset='validation'
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model_path = "/pylon5/cc5614p/deopha32/Saved_Models/cvd-model.h5"

# early_stop = EarlyStopping(monitor='val_acc',  verbose=1, patience=50)
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)
csv_logger = CSVLogger("/pylon5/cc5614p/deopha32/Saved_Models/cvd-model-history.csv", append=True)

callbacks = [checkpoint, csv_logger]

# Model Training
with tf.device('/gpu:0'):
    model = ResidualAttentionNetwork(
                input_shape=IMAGE_SHAPE, 
                n_classes=num_classes, 
                activation='softmax').build_model()
    
    model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN, verbose=0, callbacks=callbacks,
                    validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                    epochs=epochs, use_multiprocessing=True, workers=40)