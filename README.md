# Implicit-PDF
## Generate artificial training data using blenderproc
Generating the artifical images is done in the blenderproc folder

## Helper fuctions
load_image: loads hdf5 file and returns image(np.array) and ground truth(dict)
```
from lib.load_image import load_image
image, ground_truth = load_image(path)
```
