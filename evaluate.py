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

dataDir = "data/Evaluation_set/cylinder_rotation" #Add link to testing data
outputDir="output/Evaluation_Result_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +"/"

modelDir = "best_trained_weights/cylinder_weights/"
#====================SETUP==========================
files = sorted(glob.glob(dataDir + "/*.hdf5"),key=len)
random.shuffle(files)
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



evaluatePictures(model,files,outputDir,maxEvaluations=10,cutoffPercentage=0.95,resolution=50)



end_time = time.time()


#print(end_time-start_time)

