

import sys
import os
import glob
import numpy as np


from Controller import *
from load_image import *




def main() -> int:
    """Echo the input arguments to standard output"""

    dir = "../blenderproc/data"
    training = True


    files=glob.glob(dir+"/*.hdf5")

    image, ground_truth = load_image(files[0])

    img=np.array(image)
    size=(np.prod(img.shape))
    controller = Controller(size)
    x_step=0.1
    y_step=0.1
    r_step=4

    for file in files:
        image, ground_truth = load_image(file)
        print(ground_truth)
        controller.sample_space(image,x_step,y_step,r_step,truth=ground_truth,training=training)





    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit



