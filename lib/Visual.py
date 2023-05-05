import random
import math
from typing import Any

import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import time
# own libs
from lib.load_image import *
from lib.ModelArchitecture import *
from lib.Poses import *


class Heat_map:
    def save_figure(self, path):
        self.heat_figure.savefig(path)

    def __init__(self, model, figsize=(12, 6)):
        self.model = model
        self.heat_figure = plt.figure(figsize=figsize)
        self.image_ax = self.heat_figure.add_subplot(122)
        self.heat_ax = self.heat_figure.add_subplot(121, projection='3d')

    def __call__(self, path):
        self.heat_figure.show()
        image, ground_truth = load_image(path)
        images = [image]
        images = tf.convert_to_tensor(images)
        all_poses = get_all_poses(10, 10, 10)
        predictions = self.model.generate_pdf(images, all_poses)
        predictions = np.squeeze(predictions)
        predictions = predictions / np.max(predictions)
        self.heat_ax.clear()
        for i in range(np.size(predictions)):
            pose = all_poses[i]
            x_pred = pose[0]
            y_pred = pose[1]
            r_pred = math.atan2(pose[3], pose[2])
            # print(predictions[i])
            self.heat_ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=2, color="red",
                              alpha=np.clip(predictions[i] - 0.1, 0, 1))  # , label='PP')

        gt_x = float(ground_truth["x"])
        gt_y = float(ground_truth["y"])
        gt_z = float(ground_truth["r"])
        gt_z = math.radians(gt_z)
        if gt_z > math.pi:
            gt_z = gt_z - 2 * math.pi
        self.image_ax.imshow(image)
        self.image_ax.axis('off')
        self.heat_ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")  # , label='GT')
        self.heat_ax.set_xlabel("X-coor")
        self.heat_ax.set_ylabel("Y-coor")
        self.heat_ax.set_zlabel("Rotation")
        self.heat_ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
        self.heat_ax.legend()
        self.heat_figure.canvas.draw()
        self.heat_figure.canvas.flush_events()
        print("Done Plotting Heat Map")


class Plot_loss:
    def save_figure(self, path):
        self.loss_figure.savefig(path)

    def __init__(self, plot_name="Loss", plot_x_name="Epoch", plot_y_name="Loss"):
        self.name = plot_name
        self.x_name = plot_x_name
        self.y_name = plot_y_name
        self.loss_figure = plt.figure()
        self.loss_ax = self.loss_figure.add_subplot(111)

    def __call__(self, data_list_A, data_list_B=None):
        self.loss_figure.show()
        self.loss_ax.clear()
        self.loss_ax.plot(data_list_A)
        if data_list_B is not None:
            self.loss_ax.plot(data_list_B, 'ro')
        self.loss_ax.axhline(y=0.8161, color='0.8', linestyle='--')
        self.loss_ax.axhline(y=-3.7889, color='0.8', linestyle='--')
        self.loss_ax.set_xlabel(self.x_name)
        self.loss_ax.set_ylabel(self.y_name)
        self.loss_ax.set_title(self.name, fontsize=20)
        self.loss_figure.canvas.draw()
        self.loss_figure.canvas.flush_events()