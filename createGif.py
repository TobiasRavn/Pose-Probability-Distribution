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

#==================Settings======================

dataDir = "data/cylinder_rotation"

#dataDir = "data/cylinder_500"
#dataDir = "data/data_cylinder_1000"
#dataDir = "data/mini_set"
#dataDir = "data/data_cup_1000"

outputDir="output/rotationGif_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") +"/"

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


figures = Plot_the_figures(model)
count=0



#====================Create Gif Files====================
gifFiles = []
for file in files:
    print(file)
    figures(file)

    gifFile=outputDir+f"Image_{count}.png"
    figures.save_figure(gifFile)

    gifFiles.append(gifFile)
    count+=1


#====================Compile Gif====================
imageDatas=[]
for image in gifFiles:
    imageDatas.append(imageio.imread(image))
imageio.mimsave(outputDir+'animation.gif', imageDatas, loop=0, duration = 0.3)


print("done")



end_time = time.time()


#print(end_time-start_time)