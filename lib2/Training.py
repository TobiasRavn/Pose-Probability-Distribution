
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
#own libs
from lib2.load_image import *
from lib.Pose_Accumulator import *
from lib2.Descriptor import *

from lib2.ModelArchitecture import *


class Training:
    def __init__(self, dir, doEvaluation=True):


        self.doEvaluation=doEvaluation
        self.dir = dir
        files = glob.glob(self.dir + "/*.hdf5")



        random.shuffle(files)
        # split files into training and validation data
        self.train_data = files[:int(len(files) * 0.8)]
        self.vali_data = files[int(len(files) * 0.8):]

        # Set file name
        self.current_test_name = "batch_4_epoch_100_1e_4_triangle"

        # load first image to use img size
        image, ground_truth = load_image(files[0])
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
        loss_list = []
        validation_loss_list = []
        self.epochs = 100
        self.batch_size = 4

        self.modelAchitecture=ModelArchitecture(self.lenDiscriptors,self.lenPose, self.imgSize)


    @tf.function
    def compute_loss(self, images, poses, training=True):

        logits_norm = self.modelAchitecture.generate_pdf(images,poses,training)

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
        #vision_description = self.modelAchitecture.descriptor.vision_model(images, training=False)
        loss = self.compute_loss(images, poses, training=False)

        return loss



    def evaluate(self):
        pass


    def get_all_poses(self, x_num, y_num, r_num):

        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1

        step_r = 360 / r_num

        x_range = np.linspace(x_min, x_max, int(x_num))
        y_range = np.linspace(y_min, x_max, int(y_num))
        r_range = np.linspace(0, 360 - step_r, int(r_num))

        size = x_num * y_num * r_num
        allPoses = np.zeros([size, 4])
        count = 0

        for x in x_range:
            for y in y_range:
                for r in r_range:
                    r_rad = math.radians(r)

                    allPoses[count] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
                    count += 1

        return allPoses

    def get_random_poses_plus_correct(self, position_samples, ground_truth):
        poses = np.zeros((position_samples, 4))

        x = ground_truth["x"]
        y = ground_truth["y"]
        r = ground_truth["r"]
        x, y, r = float(x), float(y), float(r)

        x = x / 0.3
        y = y / 0.3
        r_rad = math.radians(r)
        poses[-1] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])

        for j in range(position_samples - 1):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            r = random.uniform(self.r_min, self.r_max)
            r_rad = math.radians(r)
            poses[j] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
        return poses

    def epochTrain(self, files):
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
            ground_truths = np.zeros((len(batch), self.position_samples, 4))
            for i, file in enumerate(batch):
                image, ground_truth = load_image(file)
                images[i] = image

                ground_truths[i] = self.get_random_poses_plus_correct(self.position_samples, ground_truth)

            # convert numpy arrays to tensors
            images = tf.convert_to_tensor(images)
            ground_truths = tf.convert_to_tensor(ground_truths)

            # time to train_step
            st = time.time()
            loss = self.train_step(self.optimizer, images, ground_truths)
            print("loss: ", loss.numpy(), " time: ", time.time() - st)


    def epochEval(self, files):

        validations_set = files
        random.shuffle(validations_set)

        batches = [validations_set[x:x + self.batch_size] for x in range(0, len(validations_set), self.batch_size)]

        validation_loses = []

        for count, batch in enumerate(batches):
            images = np.zeros((len(batch), self.imgSize[0], self.imgSize[1], self.imgSize[2]))
            # init ground truth
            ground_truths = np.zeros((len(batch), self.position_samples, 4))
            for i, file in enumerate(batch):
                image, ground_truth = load_image(file)
                images[i] = image

                ground_truths[i] = self.get_random_poses_plus_correct(self.position_samples, ground_truth)

            # convert numpy arrays to tensors
            images = tf.convert_to_tensor(images)
            ground_truths = tf.convert_to_tensor(ground_truths)

            # time to train_step
            st = time.time()

            loss = self.validation_step(images, ground_truths)
            # print("loss: ", loss.numpy(), " time: ", time.time() - st)

            validation_loses.append(loss)

        file = random.sample(self.vali_data, 1)

        image, ground_truth = load_image(file[0])
        images = [image]
        images = tf.convert_to_tensor(images)
        all_poses = self.get_all_poses(25, 25, 25)
        predictions = self.modelAchitecture.generate_pdf(images, all_poses)
        #plotHeatmap(all_poses, predictions, ground_truth, heat_fig, heat_ax, epoch_counter)

        #print("Epoch loss: ", np.mean(np.array(temp_epoch_loss)))
        #epoch_lose_list.append(np.mean(np.array(temp_epoch_loss)))
        #ax_loss_epoch.clear()
        #ax_loss_epoch.plot(epoch_lose_list)
        #ax_loss_epoch.plot(validation_loss_list, 'bo')
        #ax_loss_epoch.set_xlabel("Epoch")
        #ax_loss_epoch.set_ylabel("Loss")
        #ax_loss_epoch.set_title("Loss over Epoch", fontsize=20)
        #fig_loss_epoch.canvas.draw()
        #fig_loss_epoch.canvas.flush_events()
#
        #mlp_model.save("mlp_" + current_test_name + "_" + str(epoch_counter))
        #descriptor.vision_model.save("vm_" + current_test_name + "_" + str(epoch_counter))
        #heat_fig.savefig("new_training_output/Heat_map_" + current_test_name + "_" + str(epoch_counter) + ".png")
        #fig.savefig("new_training_output/loss_" + current_test_name + "_" + str(epoch_counter) + ".png")
        #epoch_counter += 1
        #validation_loss = np.mean(np.array(validation_loses))
        #print("Validation loss: ", validation_loss)
        #validation_loss_list.append(validation_loss)
        #

    def startTraining(self, epochs):

        print("Starting training")
        epoch_lose_list = []
        for epoch in range(epochs):
            self.epochTrain(self.train_data)

            self.epochEval(self.vali_data)

            # Validation


