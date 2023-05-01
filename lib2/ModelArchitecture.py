
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
#own libs
from lib2.load_image import *
#from Pose_Accumulator import *
from lib2.Descriptor import *

    # define class IPDF
from PIL import Image

tf.keras.utils.load_img


class ModelArchitecture:


    def __init__(self, lenDiscriptors, lenPose, imgSize):
        input_visual = tfkl.Input(shape=(lenDiscriptors,))
        visual_embedding = tfkl.Dense(256)(input_visual)
        input_query = tfkl.Input(shape=(None, lenPose,))
        query_embedding = tfkl.Dense(256)(input_query)
        output = visual_embedding[:, tf.newaxis] + query_embedding
        output = tfkl.ReLU()(output)
        output = tfkl.Dense(256, 'relu')(output)
        output = tfkl.Dense(1)(output)
        self.mlp_model = tf.keras.models.Model(
            inputs=[input_visual, input_query],
            outputs=output)

        self.descriptor = self.createDescriptor(imgSize)



    def saveModel(self, name, path="/"):
        mlp_string = path+name+"_mlp"
        vision_string = path + name + "_vis"
        self.mlp_model.save_weights(mlp_string)
        self.vision_model.save_weights(vision_string)



    def loadModel(self, name, path="/"):
        mlp_string = path + name + "_mlp"
        vision_string = path + name + "_vis"
        self.mlp_model.load_weights(mlp_string)
        self.vision_model.load_weights(vision_string)

    @tf.function
    def generate_pdf(self, images, poses, training=False):
        vision_description = self.vision_model(images, training=training)

        logits = self.mlp_model([vision_description, poses],
                                                 training=False)[Ellipsis, 0]

        logits_norm = tf.nn.softmax(logits, axis=-1)

        return logits_norm





    def get_image_descriptor_path(self, image_path):
        "This function will return a descriptor for the vision model. It takes a image path"
        _image = tf.keras.utils.load_img(image_path)
        _image = tf.keras.utils.img_to_array(_image)[:, :, :3]
        _image = np.expand_dims(_image, axis=0)
        return self.vision_model.predict(_image)

    def get_image_descriptor_array(self, image_data, image_depth="RGB"):
        "This function will return a descriptor for the vision model. It takes a hdf5"
        _image = Image.fromarray(image_data, image_depth)
        _image = tf.keras.utils.img_to_array(_image)[:, :, :3]
        _image = np.expand_dims(_image, axis=0)
        return self.vision_model.predict(_image)

    def get_image_descriptor_PIL(self, image_data):
        "This function will return a descriptor for the vision model. It takes a PIL image"
        _image = tf.keras.utils.img_to_array(image_data)[:, :, :3]
        _image = np.expand_dims(image_data, axis=0)
        return self.vision_model.predict(_image)

    def show_base_model(self):
        "Shows the base model"
        self.base_descriptor_model.summary()

    def show_vision_model(self):
        "Shows the vision model"
        self.vision_model.summary()

    def get_length_of_visual_description(self):
        return self.length_of_visual_description

    def createDescriptor(self, image_size, model_weights="imagenet"):


        tf_keras_layers = tf.keras.layers

        self.base_descriptor_model = tf.keras.applications.ResNet50V2(weights=model_weights,
                                                                      include_top=False,
                                                                      input_shape=image_size)

        self.input_image_size = tf.keras.layers.Input(shape=image_size)
        layers_added = self.base_descriptor_model(self.input_image_size)
        layers_added = tf_keras_layers.GlobalAveragePooling2D()(layers_added)

        self.length_of_visual_description = layers_added.shape[-1]
        self.vision_model = tf.keras.Model(self.input_image_size, layers_added)