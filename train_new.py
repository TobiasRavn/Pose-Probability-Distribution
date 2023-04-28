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
from lib.Pose_Accumulator import *
from lib.Descriptor import *

from lib2.Training import *

#start timer
start_time = time.time()







#dir = "blenderproc/data_500_first"

dir = "blenderproc/data"
# dir = "blenderproc/data_triangle"
# dir = "blenderproc/data_1000"

training = Training(dir)

epochs=100
training.startTraining(epochs)


end_time = time.time()

print(end_time-start_time)