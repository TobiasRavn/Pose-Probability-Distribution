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







dir = "data/cylinder_rotation"

#dir = "data/cylinder_500"
#dir = "data/data_cylinder_1000"
#dir = "data/mini_set"
#dir = "data/data_cup_1000"

outputDir="output/rotationGif_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") +"/"


files = sorted(glob.glob(dir + "/*.hdf5"),key=len)

image, ground_truth = load_image(files[0])
img = np.array(image)
imgSize = img.shape
# init descriptor

# init mlp model
lenDiscriptors = 2048
lenPose = 4



model=ModelArchitecture(lenDiscriptors,lenPose,imgSize)
model.loadModel("output/data_cup_1000_2023_05_09_01_17/","weights")
#random.shuffle(files)

os.makedirs(outputDir)
figures = Plot_the_figures(model)
count=0

import imageio
gifFiles = []
for file in files:
    print(file)
    figures(file)

    gifFile=outputDir+f"Image_{count}.png"
    figures.save_figure(gifFile)

    gifFiles.append(gifFile)
    count+=1



#with imageio.get_writer(outputDir+'animation.gif', mode='I') as writer:
#    for filename in gifFiles:
#        image = imageio.imread(filename)
#        writer.append_data(image)



imageDatas=[]
for image in gifFiles:

    imageDatas.append(imageio.imread(image))



# imageio writes all the images in a .gif file at the gif_path
imageio.mimsave(outputDir+'animation.gif', imageDatas, loop=0, duration = 0.3)


#figures = Plot_the_figures(model)
#figures(files[0],debug=True)
#pose = model.getIterativeMaxPose(files[1],25,10)



print("done")


#print(f"Prediction:   {pose[0]},\t {pose[1]},\t {pose[2]}")
#print(f"Ground Truth: {ground_truth['x']},\t {ground_truth['y']},\t {ground_truth['r']}")
#print(ground_truth)

end_time = time.time()
while(1):
    plt.show()

#print(end_time-start_time)