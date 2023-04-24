import os
import numpy as np
import h5py
import ast
import glob
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from keras.models import load_model
import sklearn as sk  
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import scipy as sp  
import tensorflow as tf  
import platform 

tf_keras_layers = tf.keras.layers

class Descriptor:
    def get_image_descriptor_array(self, image_data, image_depth = "RGB"):
        "This function will return a descriptor for the vision model. It takes a hdf5"
        _image = Image.fromarray(image_data,image_depth)
        _image = tf.keras.utils.img_to_array(_image)[:,:,:3]
        _image = np.expand_dims(_image , axis=0)
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
    def __init__(self, images, ground_truths, image_descriptor, batch_size=2, shuffle=True):
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
            #image_descriptor = Descriptor(image_size=(1000, 1000, 3))
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

class MLP:
    def __init__(self, descriptor_shape):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=descriptor_shape, name='dense_1'),  # Modify the input shape
            Dense(256, activation='relu', name='dense_2'),
            Dense(3, activation='softmax', name='predictions')
        ])

        self.descriptor_shape = descriptor_shape
        #self.lr_schedule = 1e-7

        # Learning rate scheduling
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-7, # Set your desired initial learning rate
            decay_steps=10000,
            decay_rate=0.9)

        # Define optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)            
        self.model.compile(optimizer=self.optimizer, loss='mse')

def plotHeatmap(predictions, ground_truths):

    print("Predictions:", predictions)
    print("Ground Truths:", ground_truths)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
        # Add predicted pose as a red dot
        ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=10, color="red")#, label='PP')

        # Add ground truth as a blue dot
        gt_x, gt_y, gt_z = map(float, ground_truths[i])
        ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")#, label='GT')

    # Set axis labels and titles
    ax.set_xlabel("X-coor")
    ax.set_ylabel("Y-coor")
    ax.set_zlabel("Z-coor")
    ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
    ax.legend()

     # Calculate the average distance between predicted poses and ground truth poses
    distances = []
    for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
        gt_x, gt_y, gt_z = map(float, ground_truths[i])
        distance = np.sqrt((x_pred - gt_x)**2 + (y_pred - gt_y)**2 + (r_pred - gt_z)**2)
        distances.append(distance)

    average_distance = np.mean(distances)

    # Create a unique filename for the plot
    filename = f"heatmap_{len(ground_truths)}.png"
    # Save the plot as a PNG image in the 'plots' folder
    plt.savefig(f'/Users/reventlov/Documents/Robcand/2. Semester/ProjectARC/Project/Implicit-PDF/ERH_Test/plot/{filename}')
    plt.close(fig)

    return average_distance

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

        x, y, r = output
        outputs.append((x, y, r))
        #print(f"Predicted pose: ({x}, {y}, {r})")
        #print("outputs: ", outputs)
        #input("Press any key to continue...")
    return outputs

def load_image(path):
    f = h5py.File(path, 'r')
    image = f.get('colors')[()]
    byte_str = f.get('ground_truth')[()]
    dict_str = byte_str.decode("UTF-8")
    ground_truth_dict = ast.literal_eval(dict_str)
    ground_truth = [ground_truth_dict['x'], ground_truth_dict['y'], ground_truth_dict['r']]
    f.close()
    return image, ground_truth

def main():
    
    print(f"Python Platform: {platform.platform()}")  
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
   
    device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    
    # Print the selected device
    print(f"Running on {device}")

    # Rest of your code inside the context manager
    with tf.device('/device:GPU:0'):
        dir = "/Users/reventlov/Documents/Robcand/2. Semester/ProjectARC/Project/IPDF/data1000"
        files = glob.glob(dir + "/*.hdf5")
        image_arrays = []
        ground_truths = []
        
        
        for i in range(0, len(files), 20): # only load every 40 number of file
            file = files[i]
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

        num_epochs = 15
        # Initialize a list to store the average loss for each epoch
        losses = []
        for epoch in range(num_epochs):
            #history = model.model.fit(train_dataset, epochs=1, validation_data=val_dataset)
            history = model.model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

            # Collect the training loss for the current epoch
            loss = history.history['loss'][0]
            losses.append(loss)
            print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")

        # Save the model
        model.model.save("pose_estimation_model.h5")

        predictions = predict_poses(model, val_image_arrays, image_descriptor)

        # Predict the poses for validation images
        predictions = [(x[0], x[1],  x[2]) for x in predictions]

        # Initialize a list to store the average distances
        average_distances = []
        # Load different numbers of data points and plot the heatmaps
        num_data_points_range = range(10, len(image_arrays) + 1, 10)
        for num_data_points in num_data_points_range:
            # Select a subset of data points
            subset_image_arrays = image_arrays[:num_data_points]
            subset_ground_truths = ground_truths[:num_data_points]

            # Predict the poses for the subset of images
            predictions = predict_poses(model, subset_image_arrays, image_descriptor)
            predictions = [(x[0], x[1], x[2]) for x in predictions]

            # Plot heatmap with predicted and ground truth coordinates and calculate the average distance
            average_distance = plotHeatmap(predictions, subset_ground_truths.tolist())
            average_distances.append(average_distance)

        # Plot the relationship between the number of data points and the average distance
        plt.plot(num_data_points_range, average_distances, marker='o', linestyle='-', markersize=6)
        plt.xlabel('Number of Data Points')
        plt.ylabel('Average Distance')
        plt.title('Average Distance vs. Number of Data Points')

        # Save the plot as a PNG image in the 'plots' folder
        filename1 = f"average_distance_vs_num_data_points.png"
        plt.savefig(f'/Users/reventlov/Documents/Robcand/2. Semester/ProjectARC/Project/Implicit-PDF/ERH_Test/plot/{filename1}')
        plt.close()

        # Create a list of epoch numbers
        epochs = list(range(1, len(losses) + 1))

        # Create the MSE-loss plot
        plt.plot(epochs, losses, marker='o', linestyle='-', markersize=6)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Loss vs. Epochs')

        # Save the plot as a PNG image in the 'plots' folder
        filename2 = f"mse_Epoc{num_epochs}_Loss{len(train_image_arrays) + len(val_image_arrays)}.png"

        # Save the plot as a PNG image in the 'plots' folder
        plt.savefig(f'/Users/reventlov/Documents/Robcand/2. Semester/ProjectARC/Project/Implicit-PDF/ERH_Test/plot/{filename2}')
        plt.close()

if __name__ == "__main__":
    main()
