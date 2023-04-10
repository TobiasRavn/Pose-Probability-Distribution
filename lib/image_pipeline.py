import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform


class ImageTransformer:
    def __init__(self, images, save_folder="output"):
        self.images = images
        self.save_folder = save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def _get_output_path(self, input_path):
        filename = os.path.basename(input_path)
        output_filename = os.path.splitext(filename)[0] + ".hdf5"
        output_path = os.path.join(self.save_folder, output_filename)
        return output_path

    def rotate(self, angle):
        if isinstance(self.images, Image.Image):
            # If only one image is given, convert to a list
            self.images = [self.images]

        for i, image in enumerate(self.images):
            image_array = np.array(image)
            rotated_array = np.rot90(image_array, angle//90, axes=(1, 0))
            output_path = self._get_output_path(f"image_{i}.hdf5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('colors', data=rotated_array)
                f.create_dataset('ground_truth', data="")

            print(f"Image saved to {output_path}")
            ## Display the image 
            plt.imshow(rotated_array)
            plt.show()

    def skew(self, x, y):
        if isinstance(self.images, Image.Image):
            # If only one image is given, convert to a list
            self.images = [self.images]

        for i, image in enumerate(self.images):
            image_array = np.array(image)
            skewed_array = self._skew(image_array, x, y)
            output_path = self._get_output_path(f"image_{i}.hdf5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('colors', data=skewed_array)
                f.create_dataset('ground_truth', data="")  # You may want to change this

            print(f"Image saved to {output_path}")
            ## Display the image 
            plt.imshow(skewed_array)
            plt.show()

    def _skew(self, image_array, x, y):
        width, height = image_array.shape[1], image_array.shape[0]
        max_skew = min(width, height) // 2
        x_skew = int(x * max_skew)
        y_skew = int(y * max_skew)

        if x_skew == 0 and y_skew == 0:
            return image_array

        if x_skew < 0:
            image_array = np.fliplr(image_array)
            x_skew = abs(x_skew)

        if y_skew < 0:
            image_array = np.flipud(image_array)
            y_skew = abs(y_skew)

        affine_transform_matrix = [
            1,
            x_skew / height,
            0,
            y_skew / width,
            1,
            0,
        ]

        skewed_array = Image.fromarray(image_array).transform(
            (width, height),
            Image.AFFINE,
            affine_transform_matrix,
            resample=Image.BICUBIC,
        )

        return np.array(skewed_array)
