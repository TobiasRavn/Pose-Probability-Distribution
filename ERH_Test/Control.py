import os
import numpy as np
import h5py
import ast
import glob
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from keras.models import load_model
import sklearn as sk  
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import scipy as sp  
import tensorflow as tf  
import platform 

tf_keras_layers = tf.keras.layers

class Descriptor:
    def get_image_descriptor_array(self, image_data, image_depth = "RGB"):
        "This function will return a descriptor for the vision model. It takes a hdf5"
        _image = Image.fromarray(image_data,image_depth)
        _image = tf.keras.utils.img_to_array(_image)[:,:,:3]
        _image = np.expand_dims(_image , axis=0)
        return self.vision_model.predict(_image)
    
    def show_base_model(self):
        "Shows the base model"
        self.base_descriptor_model.summary()
    
    def show_vision_model(self):
        "Shows the vision model"
        self.vision_model.summary()
        
    def get_length_of_visual_description(self):
        return self.length_of_visual_description
    
    def __init__(self, image_size, model_weights = "imagenet"):
        self.base_descriptor_model = tf.keras.applications.ResNet50V2(weights=model_weights,
                                               include_top=False,               
                                               input_shape=image_size)
        
        self.input_image_size = tf.keras.layers.Input(shape=image_size)
        layers_added = self.base_descriptor_model(self.input_image_size)
        layers_added = tf_keras_layers.GlobalAveragePooling2D()(layers_added)
        
        self.length_of_visual_description = layers_added.shape[-1]
        self.vision_model = tf.keras.Model(self.input_image_size, layers_added)

class PoseEstimationDataset(keras.utils.Sequence):
    def __init__(self, images, ground_truths, image_descriptor, batch_size=2, shuffle=True):
        self.images = images
        self.ground_truths = ground_truths
        self.image_descriptor = image_descriptor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))  # Add this line to initialize 'indexes'
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        # Generate descriptors
        X_descriptors = []
        for image in X:
            image_descriptor_array = self.image_descriptor.get_image_descriptor_array(image)
            X_descriptors.append(image_descriptor_array)

        return np.array(X_descriptors), np.array(y)
    
    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        # Initialization
        X = np.empty((self.batch_size, *self.images.shape[1:]))
        y = np.empty((self.batch_size, *self.ground_truths.shape[1:]))

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i,] = self.images[index]

            # Store ground truth
            y[i,] = self.ground_truths[index
            # Apply random horizontal flip with probability p_flip
            if self.p_flip and np.random.rand() < self.p_flip:
                X[i,] = np.fliplr(X[i,])
                y[i,] = np.fliplr(y[i,])

            # Apply random rotation with probability p_rot
            if self.p_rot and np.random.rand() < self.p_rot:
                angle = np.random.uniform(-self.max_rot_angle, self.max_rot_angle)
                X[i,] = rotate(X[i,], angle, mode='reflect')
                y[i,] = rotate(y[i,], angle, mode='reflect')

            # Apply random zoom with probability p_zoom
            if self.p_zoom and np.random.rand() < self.p_zoom:
                zoom_factor = np.random.uniform(1 - self.max_zoom, 1 + self.max_zoom)
                X[i,] = zoom(X[i,], zoom_factor, mode='reflect')
                y[i,] = zoom(y[i,], zoom_factor, mode='reflect')

        return X, y
        
class AugmentedPoseEstimationDataset(PoseEstimationDataset):
    def init(self, images, ground_truths, image_descriptor, batch_size=2, shuffle=True,
    p_flip=0.5, p_rot=0.5, max_rot_angle=10, p_zoom=0.5, max_zoom=0.1):
    super().init(images, ground_truths, image_descriptor, batch_size=batch_size, shuffle=shuffle)
    self.p_flip = p_flip
    self.p_rot = p_rot
    self.max_rot_angle = max_rot_angle
    self.p_zoom = p_zoom
    self.max_zoom = max_zoom
    
    def __data_generation(self, indexes):
    """
    Generates data containing batch_size samples with augmentations.
    """
    # Initialization
    X = np.empty((self.batch_size, *self.images.shape[1:]))
    y = np.empty((self.batch_size, *self.ground_truths.shape[1:]))

    # Generate data
    for i, index in enumerate(indexes):
        # Store sample
        X[i,] = self.images[index]

        # Store ground truth
        y[i,] = self.ground_truths[index]

        # Apply random horizontal flip with probability p_flip
        if self.p_flip and np.random.rand() < self.p_flip:
            X[i,] = np.fliplr(X[i,])
            y[i,] = np.fliplr(y[i,])

        # Apply random rotation with probability p_rot
        if self.p_rot and np.random.rand() < self.p_rot:
            angle = np.random.uniform(-self.max_rot_angle, self.max_rot_angle)
            X[i,] = rotate(X[i,], angle, mode='reflect')
            y[i,] = rotate(y[i,], angle, mode='reflect')

        # Apply random zoom with probability p_zoom
        if self.p_zoom and np.random.rand() < self.p_zoom:
            zoom_factor = np.random.uniform(1 - self.max_zoom, 1 + self.max_zoom)
            X[i,] = zoom(X[i,], zoom_factor, mode='reflect')
            y[i,] = zoom(y[i,], zoom_factor, mode='reflect')

    return X, y
