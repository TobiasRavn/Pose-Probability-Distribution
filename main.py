import sys
import os
import glob
import numpy as np


from lib.Controller import *
from lib.load_image import *
from lib.image_pipeline import *




def main() -> int:
    """Echo the input arguments to standard output"""

    dir = "blenderproc/data"
    training = True

    files=glob.glob(dir+"/*.hdf5")
    image, ground_truth = load_image(files[0])

    # ## Used for testing image rotation through image_pipeline class
    # transformer = ImageTransformer(image, save_folder="output")
    # transformer.rotate(90)
    # ## Skew all images by (0.2, 0.3) and save to output_skew folder
    # transformer.skew(x=0.2, y=0.3)

    # transformer = ImageTransformer(image)
    # transformer.salt_and_pepper_noise(50)  # Add 10% salt and pepper noise to all images

    img=np.array(image)
    size=img.shape
    controller = Controller(size)
    x_step=0.1
    y_step=0.1
    r_step=4

    epochs=10
    for epoch in range(epochs):
        for count, file in enumerate(files):
            image, ground_truth = load_image(file)
            #print(ground_truth)
            print("file: ",count,"/",len(files) , " epoch: ",epoch,"/", epochs)
            controller.sample_space_all_at_once(image,x_step,y_step,r_step,truth=ground_truth,training=training)


    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit



