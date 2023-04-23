import os
import numpy as np
import h5py
import ast
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Flatten

from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

tf_keras_layers = tf.keras.layers


class Descriptor:
    def get_image_descriptor_path(self, image_path):
        "This function will return a descriptor for the vision model. It takes a image path"
        _image = tf.keras.utils.load_img(image_path)
        _image = tf.keras.utils.img_to_array(_image)[:,:,:3]
        _image = np.expand_dims(_image , axis=0)
        return self.vision_model.predict(_image)
    
    def get_image_descriptor_array(self, image_data, image_depth = "RGB"):
        "This function will return a descriptor for the vision model. It takes a hdf5"
        _image = Image.fromarray(image_data,image_depth)
        _image = tf.keras.utils.img_to_array(_image)[:,:,:3]
        _image = np.expand_dims(_image , axis=0)
        return self.vision_model.predict(_image)
    
    def get_image_descriptor_PIL(self, image_data):
        "This function will return a descriptor for the vision model. It takes a PIL image"
        _image = tf.keras.utils.img_to_array(image_data)[:,:,:3]
        _image = np.expand_dims(image_data , axis=0)
        return self.vision_model.predict(_image)
        
    def show_base_model(self):
        "Shows the base model"
        self.base_descriptor_model.summary()
    
    def show_vision_model(self):
        "Shows the vision model"
        self.vision_model.summary()
        
    def get_length_of_visual_description(self):
        return self.length_of_visual_description
    
    def __init__(self, image_size, model_weights = "imagenet"):
        self.base_descriptor_model = tf.keras.applications.ResNet50V2(weights=model_weights,
                                               include_top=False,               
                                               input_shape=image_size)
        
        self.input_image_size = tf.keras.layers.Input(shape=image_size)
        layers_added = self.base_descriptor_model(self.input_image_size)
        layers_added = tf_keras_layers.GlobalAveragePooling2D()(layers_added)
        
        self.length_of_visual_description = layers_added.shape[-1]
        self.vision_model = tf.keras.Model(self.input_image_size, layers_added)

class PoseEstimationDataset(keras.utils.Sequence):
    def __init__(self, images, ground_truths, image_descriptor, batch_size=1, shuffle=True):
        self.images = images
        self.ground_truths = ground_truths
        self.image_descriptor = image_descriptor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))  # Add this line to initialize 'indexes'
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        # Generate descriptors
        X_descriptors = []
        for image in X:
            # Remove the line below
            # image_descriptor = Descriptor(image_size=(1000, 1000, 3))
            image_descriptor_array = self.image_descriptor.get_image_descriptor_array(image)
            X_descriptors.append(image_descriptor_array)

        return np.array(X_descriptors), np.array(y)
    
    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        # Initialization
        X = np.empty((self.batch_size, *self.images.shape[1:]))
        y = np.empty((self.batch_size, *self.ground_truths.shape[1:]))

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i,] = self.images[index]

            # Store ground truth
            y[i,] = self.ground_truths[index]

        return X, y


# class MLP:
#     def __init__(self, descriptor_shape):
#         self.model = Sequential([
#             Flatten(input_shape=descriptor_shape),  # Add Flatten layer to reshape the input
#             Dense(256, activation='relu', name='dense_1'),
#             Dense(256, activation='relu', name='dense_2'),
#             Dense(2, activation='softmax', name='predictions')
#         ])

#         self.descriptor_shape = descriptor_shape
#         self.learning_rate = 1e-6

#         # Define optimizer and loss function
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
#         self.model.compile(optimizer=self.optimizer, loss='mse')

class MLP:
    def __init__(self, descriptor_shape):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=descriptor_shape, name='dense_1'),  # Modify the input shape
            Dense(256, activation='relu', name='dense_2'),
            Dense(2, activation='softmax', name='predictions')
        ])

        self.descriptor_shape = descriptor_shape
        self.learning_rate = 1e-6

        # Define optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='mse')


def plotHeatmap(predictions, ground_truths):
    # Construct covariance matrix
    var_x = 0.1
    var_y = 0.1
    corr_xy = 0.5
    cov_matrix = np.array([[var_x, corr_xy*np.sqrt(var_x*var_y)],
                           [corr_xy*np.sqrt(var_x*var_y), var_y]])

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (x_pred, y_pred) in enumerate(predictions):
        # Calculate probabilities using multivariate Gaussian distribution
        xy = np.vstack((x_pred, y_pred)).T
        prob = np.exp(-0.5 * np.sum(xy.dot(np.linalg.inv(cov_matrix)) * xy, axis=1))

        # Create a scatter plot with a colorbar
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=1)
        colors = cmap(norm(prob))

        sc = ax.scatter(x_pred, y_pred, c=colors, cmap=cmap, s=20, edgecolors="black")

        # Add predicted pose as a red dot and ground truth as a blue dot
        ax.plot(x_pred, y_pred, marker='o', markersize=10, color="red")
        #ax.plot(ground_truths[i][0], ground_truths[i][1], marker='o', markersize=10, color="blue")
        ax.plot([ground_truths[i][0]], [ground_truths[i][1]], marker='o', markersize=10, color="blue")


    # Move colorbar to the right side
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, shrink=0.7, aspect=10, 
                        fraction=0.15, label='Probability')
    cbar.ax.tick_params(labelsize=10)
    cax = cbar.ax
    cax.yaxis.set_label_position('right')
    cax.yaxis.set_ticks_position('right')
    cax.set_ylabel('Probability', rotation=-90, va='bottom', fontsize=14, labelpad=10)

    # Set axis labels and titles
    ax.set_xlabel("X-coor")
    ax.set_ylabel("Y-coor")
    ax.set_title("Probability of Pose Estimation of Cup in One Plane", fontsize=20)

    plt.show()


# def predict_poses(model, image_arrays, transform):
#     outputs = []
#     for image_array in image_arrays:
#         image = Image.fromarray(image_array)
#         if transform:
#             image = transform(image)
#         #image = np.array(image).reshape(1, -1)
#         image = np.array(image)
#         # Assuming 'input_data' is your input data with shape (None, 3000000)   
#         image = tf.reshape(image, (-1, 1, 2048))
#         output = model.model.predict(image)
#         x, y = output[0]
#         outputs.append((x, y))
#     return outputs

# def predict_poses(model, image_arrays, image_descriptor):
#     outputs = []
#     for image_array in image_arrays:
#         # Get the image descriptor
#         image_descriptor_array = image_descriptor.get_image_descriptor_array(image_array)

#         # Use the image descriptor as input to the model
#         output = model.model.predict(image_descriptor_array)

#         x, y = output[0]
#         outputs.append((x, y))
#     return outputs

def predict_poses(model, image_arrays, image_descriptor):
    outputs = []
    for image_array in image_arrays:
        # Get the image descriptor
        image_descriptor_array = image_descriptor.get_image_descriptor_array(image_array)

        # Add a batch dimension to the image descriptor array
        image_descriptor_array = np.expand_dims(image_descriptor_array, axis=0)

        # Use the image descriptor as input to the model
        output = model.model.predict(image_descriptor_array)

        # Remove extra dimensions
        output = np.squeeze(output)

        x, y = output
        outputs.append((x, y))
    return outputs







def load_image(path):
    f = h5py.File(path, 'r')
    image = f.get('colors')[()]
    byte_str = f.get('ground_truth')[()]
    dict_str = byte_str.decode("UTF-8")
    ground_truth_dict = ast.literal_eval(dict_str)
    ground_truth = [ground_truth_dict['x'], ground_truth_dict['y']]
    f.close()
    return image, ground_truth

def main():

    # Set the device you want to use
    device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    
    # Print the selected device
    print(f"Running on {device}")

    # Rest of your code inside the context manager
    with tf.device(device):
            
        dir = "/home/reventlov/RobCand/2. Semester/Project_AR/IPDF/data25"
        files = glob.glob(dir + "/*.hdf5")

        image_arrays = []
        ground_truths = []

        for file in files:
            image, ground_truth = load_image(file)
            image_arrays.append(image)
            ground_truths.append(ground_truth)

        image_arrays = np.array(image_arrays)
        ground_truths = np.array(ground_truths)

        # Split the dataset into training and validation sets
        train_image_arrays, val_image_arrays, train_ground_truths, val_ground_truths = train_test_split(image_arrays, ground_truths, test_size=0.1, random_state=42)

        image_descriptor = Descriptor(image_size=(1000, 1000, 3))

        print(f"Number of training images: {len(train_image_arrays)}")
        print(f"Number of validation images: {len(val_image_arrays)}")

        input("Press any key to continue...")

        descriptor_shape = (1, image_descriptor.get_length_of_visual_description())

        # Prepare the dataset
        train_dataset = PoseEstimationDataset(train_image_arrays, train_ground_truths, image_descriptor)    
        val_dataset = PoseEstimationDataset(val_image_arrays, val_ground_truths, image_descriptor)

        model = MLP(descriptor_shape)

        model.model.summary()

        # Print the descriptor shape
        print("Descriptor shape:", descriptor_shape)


        num_epochs = 10
        model.model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

        # Save the model
        model.model.save("pose_estimation_model.h5")

        # Load the saved model
        saved_model = load_model("pose_estimation_model.h5")

        # Print the summary of the loaded model
        saved_model.summary()

        predictions = predict_poses(model, val_image_arrays, image_descriptor)

        # Predict the poses for validation images
        #predictions = predict_poses(model, val_image_arrays, None)
        predictions = [(x[0], x[1]) for x in predictions]

        # Compute errors
        # ... (same as before) ...

        # Plot heatmap with predicted and ground truth coordinates
        plotHeatmap(predictions, val_ground_truths.tolist())

if __name__ == "__main__":
    main()