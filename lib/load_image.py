import h5py
import ast
from PIL import Image

#function to load image and ground truth
def load_image(path):
    f = h5py.File(path, 'r')
    image = f.get('colors')[()]
    byte_str = f.get('ground_truth')[()]
    dict_str = byte_str.decode("UTF-8")
    ground_truth = ast.literal_eval(dict_str)
    f.close()
    ## Used for image_pipeline class
    #image = Image.fromarray(image, mode='RGB')
    return image, ground_truth