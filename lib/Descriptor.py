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
        self.base_descriptor_model = tf.keras.applications.ResNet101V2(weights=model_weights,
                                               include_top=False,               
                                               input_shape=image_size)
        
        self.input_image_size = tf.keras.layers.Input(shape=image_size)
        layers_added = self.base_descriptor_model(self.input_image_size)
        layers_added = tf_keras_layers.GlobalAveragePooling2D()(layers_added)
        
        self.length_of_visual_description = layers_added.shape[-1]
        self.vision_model = tf.keras.Model(self.input_image_size, layers_added)

class tranining_model:
    def show_MLP_model(self):
        self.implicit_model.summary()
        
    def return_model(self):
        return self.implicit_model
    
    
    def __init__(self, lenght_visual_description,MLP_layer_size, number_fourier_components = 0) -> None:
        self.len_visual_description = lenght_visual_description
        #self.len_rotation = 9 #Orignal
        self.len_rotation = 3
        self.fourier = number_fourier_components
        
        if number_fourier_components == 0:
            self.length_query = self.len_rotation
        else:
            self.length_query = self.len_rotation * self.fourier
        
        
        input_visual = tf_keras_layers.Input(shape=(lenght_visual_description))
        visual_embedding = tf_keras_layers.Dense(MLP_layer_size[0])(input_visual)
        input_query = tf_keras_layers.Input(shape=(None, self.length_query,))
        query_embedding = tf_keras_layers.Dense(MLP_layer_size[0])(input_query)
        
        output = visual_embedding[:, tf.newaxis] + query_embedding
        
        output = tf_keras_layers.ReLU()(output)
        
        for num_units in MLP_layer_size[1:]:
            output = tf_keras_layers.Dense(num_units, 'relu')(output)
        output = tf_keras_layers.Dense(1)(output)  #Need to look into why it works
        self.implicit_model = tf.keras.models.Model(
            inputs=[input_visual, input_query],
            outputs=output)
        self.MPL_layer_size = MLP_layer_size
        
        


desc_test = Descriptor((255,255,3))

desc_test.show_base_model()
desc_test.show_vision_model()

disc_output = desc_test.get_image_descriptor("lib/test_images/png_dog_smol.png")
disc_vis_ou = desc_test.get_length_of_visual_description()
print(disc_output)
print(disc_vis_ou)

train_test = tranining_model(disc_vis_ou,[255]*2)


train_test.show_MLP_model()


the_train_model = train_test.return_model()

