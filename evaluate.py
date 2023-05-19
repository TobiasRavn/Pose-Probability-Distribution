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
import datetime
from lib.Visual import *
import imageio
from lib.Evaluater import *
from lib.Poses import *
#==================Settings======================

dataDir = "data/data_500_first"

#dataDir = "data/cylinder_500"
#dataDir = "data/data_cylinder_1000"
#dataDir = "data/mini_set"
#dataDir = "data/data_cup_1000"

outputDir="output/Evaluation_Result_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +"/"

modelDir= "output/data_cup_1000_2023_05_09_01_17/"


#====================SETUP==========================
files = sorted(glob.glob(dataDir + "/*.hdf5"),key=len)
os.makedirs(outputDir)



image, ground_truth = load_image(files[0])
img = np.array(image)
imgSize = img.shape
# init descriptor

# init mlp model
lenDiscriptors = 2048
lenPose = 4




model=ModelArchitecture(lenDiscriptors,lenPose,imgSize)
model.loadModel(modelDir,"weights")
#random.shuffle(files)
poses = get_random_poses_plus_correct(10,ground_truth)

print(poses)

x = ground_truth["x"]
y = ground_truth["y"]
r = ground_truth["r"]
x, y, r = float(x), float(y), float(r)

x,y= normalizePos(x,y)
r_rad = math.radians(r)


count=0


#fullEvaluationImage(model, files[0], outputDir)



evaluatePictures(model,files,outputDir,maxEvaluations=4,cutoffPercentage=0.99,resolution=50)



end_time = time.time()


#print(end_time-start_time)

