import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        all_poses = get_all_poses(25, 25, 25)
        predictions = self.model.generate_pdf(images, all_poses)
        predictions = np.squeeze(predictions)
        predictions = predictions / np.max(predictions)
        self.heat_ax.clear()
        self.image_ax.clear()
        for i in range(np.size(predictions)):
            pose = all_poses[i]
            x_pred = pose[0]
            y_pred = pose[1]
            r_pred = math.atan2(pose[3], pose[2])
            x_pred, y_pred = unnormalizePose(x_pred, y_pred)
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
        self.image_ax.set_title("Object in heatmap", fontsize=20)
        self.image_ax.axis('off')
        self.heat_ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")  # , label='GT')
        self.heat_ax.set_xlabel("X-coor")
        self.heat_ax.set_ylabel("Y-coor")
        self.heat_ax.set_zlabel("Rotation")
        self.heat_ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
        self.heat_figure.canvas.draw()
        self.heat_figure.canvas.flush_events()
        print("Done Plotting Heat Map")

class Plot_the_figures:
    def save_figure(self, path):
        self.figure.savefig(path)
        
    def __init__(self, model, figsize=(12, 12)):
        self.model = model
        self.figure = plt.figure(figsize=figsize)
        self.figure_2D_ax_xy    = self.figure.add_subplot(221)
        self.figure_2D_ax_image = self.figure.add_subplot(222)
        self.figure_3D_ax_heat  = self.figure.add_subplot(223, projection='3d')
        self.figure_2D_ax_rot   = self.figure.add_subplot(224)
        
    
    def __call__(self, path):
        self.figure.show()
        image, ground_truth = load_image(path)
        images = [image]
        images = tf.convert_to_tensor(images)
        all_poses = get_all_poses(25, 25, 25)
        predictions = self.model.generate_pdf(images, all_poses)
        predictions = np.squeeze(predictions)
        predictions = predictions / np.max(predictions)
        self.figure_2D_ax_xy.clear()
        self.figure_2D_ax_image.clear()
        self.figure_3D_ax_heat.clear()
        self.figure_2D_ax_rot.clear()
        self.figure_2D_ax_xy.set_title("Top View", fontsize=20)
        self.figure_2D_ax_xy.set_xlabel("X")
        self.figure_2D_ax_xy.set_ylabel("Y")
        self.figure_3D_ax_heat.set_title("3D Heatmap", fontsize=20)
        self.figure_3D_ax_heat.set_xlabel("X")
        self.figure_3D_ax_heat.set_ylabel("Y")
        self.figure_3D_ax_heat.set_zlabel("Rotation")
        self.figure_2D_ax_rot.set_title("Side View", fontsize=20)
        self.figure_2D_ax_rot.set_xlabel("Y")
        self.figure_2D_ax_rot.set_ylabel("Rotation")
        for i in range(np.size(predictions)):
            #print("Prediction: ", predictions[i])
            pose = all_poses[i]
            x_pred = pose[0]
            y_pred = pose[1]
            r_pred = math.atan2(pose[3], pose[2])
            x_pred, y_pred = unnormalizePose(x_pred, y_pred)
            self.figure_2D_ax_xy.plot([x_pred], [y_pred], marker='o', markersize=2, color="red",
                              alpha=np.clip(predictions[i] - 0.1, 0, 1))  # , label='PP')
            self.figure_3D_ax_heat.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=2, color="red",
                              alpha=np.clip(predictions[i] - 0.1, 0, 1))  # , label='PP')
            self.figure_2D_ax_rot.plot([y_pred], [r_pred], marker='o', markersize=2, color="red",
                              alpha=np.clip(predictions[i] - 0.1, 0, 1))  # , label='PP')
        gt_x = float(ground_truth["x"])
        gt_y = float(ground_truth["y"])
        gt_z = float(ground_truth["r"])
        gt_z = math.radians(gt_z)
        if gt_z > math.pi:
            gt_z = gt_z - 2 * math.pi
            
        pose_guess_vector =  self.model.getIterativeMaxPose(path, 25)
        x_pred = pose_guess_vector[0]
        y_pred = pose_guess_vector[1]
        r_pred = math.radians(pose_guess_vector[2])
        self.figure_2D_ax_xy.plot([x_pred], [y_pred], marker='o', markersize=10, color="blue")  # , label='GT')
        self.figure_3D_ax_heat.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=10, color="blue")  # , label='GT')
        self.figure_2D_ax_rot.plot([y_pred], [r_pred], marker='o', markersize=10, color="blue")  # , label='GT')
        #getIterativeMaxPose(self, imagePath, resolution, depth=10, zoomFactor=0.5):
        #getMaxPose(self, images , x_num, y_num, r_num, xmin=-0.3, xmax=0.3, ymin=-0.3, ymax=0.3, rmin=0,rmax=360, training=False):
        self.figure_2D_ax_xy.plot([gt_x], [gt_y], marker='o', markersize=10, color="green")  # , label='GT')
        self.figure_3D_ax_heat.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="green")  # , label='GT')
        self.figure_2D_ax_rot.plot([gt_y], [gt_z], marker='o', markersize=10, color="green")  # , label='GT')
        self.figure_2D_ax_image.imshow(image)
        self.figure_2D_ax_image.set_title("Image", fontsize=20)
        self.figure_2D_ax_image.axis('off')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        print("Done Plotting 2D plots")

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