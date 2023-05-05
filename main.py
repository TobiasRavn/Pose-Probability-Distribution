import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
#own libs
from lib.load_image import *
from lib.Descriptor import *

from lib.Training import *
from lib.ModelArchitecture import *
#start timer
start_time = time.time()







#dir = "blenderproc/data_500_first"

#dir = "data/cylinder_500"
dir = "data/mini_set"
# dir = "blenderproc/data_triangle"
# dir = "blenderproc/data_1000"

#training = Training(dir)

epochs=10
#training.startTraining(epochs)

files = glob.glob(dir + "/*.hdf5")

image, ground_truth = load_image(files[0])
img = np.array(image)
imgSize = img.shape
# init descriptor

# init mlp model
lenDiscriptors = 2048
lenPose = 4

model=ModelArchitecture(lenDiscriptors,lenPose,imgSize)
model.loadModel("output/mini_set_2023_05_05_15_09/","model")
random.shuffle(files)
pose = model.getIterativeMaxPose(files[1],10,10)

print('{}, {}, {}'.format(pose[0],pose[1],pose[2]))
print(ground_truth)

end_time = time.time()

print(end_time-start_time)