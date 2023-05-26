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

from lib.Training import *
from lib.ModelArchitecture import *
#start timer
start_time = time.time()

from lib.Visual import *

#==================Train model======================#
#  Below is the training comands.                   #
#  set the dir to the directory of images and start #
#===================================================#
dir = "data/data_triangle_1000"

training = Training(dir)       
epochs=100                     
training.startTraining(epochs) 



#==================Testing code=====================#
#  Below is some legacy code                        #
#  Use evaluate.py instead                          #
#===================================================#
#files = glob.glob(dir + "/*.hdf5")
#
#image, ground_truth = load_image(files[0])
#img = np.array(image)
#imgSize = img.shape
## init descriptor
#
## init mlp model
#lenDiscriptors = 2048
#lenPose = 4
#
#model=ModelArchitecture(lenDiscriptors,lenPose,imgSize)
#model.loadModel("best_trained_weights/","weights")   #First define the path to the weights, then what they are called
#random.shuffle(files)
#
#figures = Plot_the_figures(model)
#figures(files[0],debug=True)
#pose = model.getIterativeMaxPose(files[1],25,10)
#
##print(f"Prediction:   {pose[0]},\t {pose[1]},\t {pose[2]}")
#print(f"Ground Truth: {ground_truth['x']},\t {ground_truth['y']},\t {ground_truth['r']}")
#print(ground_truth)

end_time = time.time()
#while(1):
#    plt.show()

#print(end_time-start_time)