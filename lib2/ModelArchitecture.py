
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

        self.descriptor = Descriptor(imgSize)

    @tf.function
    def generate_pdf(self, images, poses, training=False):
        vision_description = self.descriptor.vision_model(images, training=training)

        logits = self.mlp_model([vision_description, poses],
                                                 training=False)[Ellipsis, 0]

        logits_norm = tf.nn.softmax(logits, axis=-1)

        return logits_norm