from PIL import Image

class ImageTools:
    def show_image_array(self, image_array, name=None, image_depth_type="RGB"):
        img = Image.fromarray(image_array, image_depth_type)
        img.show(name)

    def show_image_path(self, image_path, name=None):
        img = Image.open(image_path)
        img.show(name)

    def resize_image_array(self, image_array, new_size, image_depth_type="RGB"):
        "Takes a array of the image and new size takes a (X,Y) tuple, image depth is what ever type is needed."
        img = Image.fromarray(image_array, image_depth_type)
        img = img.resize(new_size)
        return img

    def resize_image_path(self, image_path, new_size):
        "Takes a path to the image and new size takes a (X,Y) tuple, image depth is what ever type is needed."
        img = Image.open(image_path)
        img = img.resize(new_size)
        return img 

    def save_image_array(self, image_array, save_path, size=None, image_depth_type="RGB"):
        img = Image.fromarray(image_array, image_depth_type)
        if size is not None:
            img = img.resize(size)
        img.save(save_path)
