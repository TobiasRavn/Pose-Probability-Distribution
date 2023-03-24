#define class IPDF

import numpy as np
import tensorflow as tf
from keras import applications

tf.keras.utils.load_img

tf_keras_layers = tf.keras.layers

class Descriptor:
    def get_image_descriptor(self, image_path):
        "This function will return a descriptor for the vision model. It takes a image path"
        _image = tf.keras.utils.load_img(image_path)
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
