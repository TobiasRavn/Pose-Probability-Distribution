import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob

tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
# own libs
from lib.load_image import *
from lib.Descriptor import *
from lib.Poses import *

from lib.ModelArchitecture import *
from lib.Visual import *


class Training:
    def __init__(self, dir, doEvaluation=True):

        self.doEvaluation = doEvaluation
        self.dir = dir
        files = glob.glob(self.dir + "/*.hdf5")

        random.shuffle(files)
        # split files into training and validation data
        self.train_data = files[:int(len(files) * 0.8)]
        self.vali_data = files[int(len(files) * 0.8):]

        # Set file name
        self.current_test_name = "batch_4_epoch_100_1e_4_triangle"

        # load first image to use img size
        image, self.ground_truth = load_image(files[0])
        img = np.array(image)
        self.imgSize = img.shape
        # init descriptor

        # init mlp model
        self.lenDiscriptors = 2048
        self.lenPose = 4
        self.learning_rate = 1e-4

        # define MLP from IPDF

        # define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.x_min, self.x_max = -0.3, 0.3
        self.y_min, self.y_max = -0.3, 0.3
        self.r_min, self.r_max = 0, 360

        self.position_samples = 100
        self.epochs = 100
        self.batch_size = 4

        self.modelAchitecture = ModelArchitecture(self.lenDiscriptors, self.lenPose, self.imgSize)

        self.prediction = None
        self.loss_list = []
        self.vali_loss = []
        self.epoch_loss = []
        self.validation_loss_epoch = []

        self.debug_loss = Plot_loss()

    def compute_loss(self, images, poses, training=True):

        logits_norm = self.modelAchitecture.generate_pdf(images, poses, training)
        print(f"Logifts norm: {logits_norm[:, -1]}")
        loss_value = -tf.reduce_mean(tf.math.log(logits_norm[:, -1] / (
                ((0.6 ** 2) * 3.1415 * 2) / poses.shape[1])))  # index -1 because last one is the correct pose
        return loss_value

    @tf.function
    def train_step(self, optimizer, images, poses):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(images, poses)
        grads = tape.gradient(
            loss,
            self.modelAchitecture.vision_model.trainable_variables + self.modelAchitecture.mlp_model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.modelAchitecture.vision_model.trainable_variables +
                self.modelAchitecture.mlp_model.trainable_variables))
        return loss

    def validation_step(self, images, poses):
        # vision_description = self.modelAchitecture.descriptor.vision_model(images, training=False)
        loss = self.compute_loss(images, poses, training=False)

        return loss

    def evaluate(self, images):
        poses = self.get_all_poses(100, 100, 100)

        logLikelihood = -self.compute_loss(images, poses, False)

        return logLikelihood

    def epochTrain(self, files, debug=False):
        temp_epoch_loss = []

        #    file = random.sample(vali_data, 1)
        #
        #    image, ground_truth = load_image(file[0])
        #    images=[image]
        #    images = tf.convert_to_tensor(images)
        #    all_poses = get_all_poses(0.05, 0.05, 10)
        #    predictions = generate_pdf(descriptor.vision_model, mlp_model, images, all_poses)
        #    plotHeatmap(all_poses, predictions, ground_truth, heat_fig,heat_ax, epoch_counter)

        random.shuffle(files)
        batches = [files[x:x + self.batch_size] for x in range(0, len(files), self.batch_size)]
        for count, batch in enumerate(batches):

            # dont use batch size here because the last one might be smaller
            # init images
            images = np.zeros((len(batch), self.imgSize[0], self.imgSize[1], self.imgSize[2]))
            # init ground truth
            poses = np.zeros((len(batch), self.position_samples, 4))
            for i, file in enumerate(batch):
                image, ground_truth = load_image(file)
                images[i] = image

                poses[i] = get_random_poses_plus_correct(self.position_samples, ground_truth)

            # convert numpy arrays to tensors
            images = tf.convert_to_tensor(images)
            poses = tf.convert_to_tensor(poses)

            # time to train_step
            st = time.time()
            loss = self.train_step(self.optimizer, images, poses)
            self.loss_list.append(loss)
            temp_epoch_loss.append(loss)
            if debug:
                self.debug_loss(self.loss_list)
            print("loss: ", loss.numpy(), " time: ", time.time() - st)
        self.epoch_loss.append(np.mean(temp_epoch_loss))

    def epochEval(self, files):

        validations_set = files
        random.shuffle(validations_set)

        batches = [validations_set[x:x + self.batch_size] for x in range(0, len(validations_set), self.batch_size)]
        temp_validation_loses = []

        for count, batch in enumerate(batches):
            images = np.zeros((len(batch), self.imgSize[0], self.imgSize[1], self.imgSize[2]))
            # init ground truth
            ground_truths = np.zeros((len(batch), self.position_samples, 4))
            for i, file in enumerate(batch):
                image, ground_truth = load_image(file)
                images[i] = image

                ground_truths[i] = get_random_poses_plus_correct(self.position_samples, ground_truth)

            # convert numpy arrays to tensors
            images = tf.convert_to_tensor(images)
            ground_truths = tf.convert_to_tensor(ground_truths)

            # time to train_step
            st = time.time()

            loss = self.validation_step(images, ground_truths)
            # print("loss: ", loss.numpy(), " time: ", time.time() - st)

            temp_validation_loses.append(loss)
            self.vali_loss.append(loss)
        self.validation_loss_epoch.append(np.mean(temp_validation_loses))
        file = random.sample(self.vali_data, 1)

        image, self.ground_truth = load_image(file[0])
        images = [image]
        images = tf.convert_to_tensor(images)
        self.all_poses = get_all_poses(25, 25, 25)
        self.predictions = self.modelAchitecture.generate_pdf(images, self.all_poses)
        # plotHeatmap(all_poses, predictions, ground_truth, heat_fig, heat_ax, epoch_counter)

        # print("Epoch loss: ", np.mean(np.array(temp_epoch_loss)))
        # epoch_lose_list.append(np.mean(np.array(temp_epoch_loss)))
        # ax_loss_epoch.clear()
        # ax_loss_epoch.plot(epoch_lose_list)
        # ax_loss_epoch.plot(validation_loss_list, 'bo')
        # ax_loss_epoch.set_xlabel("Epoch")
        # ax_loss_epoch.set_ylabel("Loss")
        # ax_loss_epoch.set_title("Loss over Epoch", fontsize=20)
        # fig_loss_epoch.canvas.draw()
        # fig_loss_epoch.canvas.flush_events()

    #
    # mlp_model.save("mlp_" + current_test_name + "_" + str(epoch_counter))
    # descriptor.vision_model.save("vm_" + current_test_name + "_" + str(epoch_counter))
    # heat_fig.savefig("new_training_output/Heat_map_" + current_test_name + "_" + str(epoch_counter) + ".png")
    # fig.savefig("new_training_output/loss_" + current_test_name + "_" + str(epoch_counter) + ".png")
    # epoch_counter += 1
    # validation_loss = np.mean(np.array(validation_loses))
    # print("Validation loss: ", validation_loss)
    # validation_loss_list.append(validation_loss)
    #

    def startTraining(self, epochs):
        print("Starting training")
        epoch_lose_list = []
        heat_map = Heat_map(self.modelAchitecture)
        plot_loss = Plot_loss()

        for epoch in range(epochs):
            self.epochTrain(self.train_data, True)
            self.epochEval(self.vali_data)
            # plot_loss(self.epoch_loss,self.validation_loss_epoch)
            heat_map(self.vali_data[0])
            # Validation
        plot_loss.save_figure("output/loss.png")

