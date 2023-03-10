#import function from load_image.py
from lib.load_image import load_image
import cv2

#load image and ground truth
image, ground_truth = load_image("blenderproc/data/1.hdf5")
print(ground_truth)
print(ground_truth.get("x"))
print(ground_truth.get("y"))
print(ground_truth.get("r"))

#show image
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()