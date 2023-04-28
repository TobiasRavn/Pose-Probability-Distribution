
import random
import math

import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import time
#own libs
from lib.load_image import *



class Visual:
    def __init__(self):
        pass


    def plotHeatmap(poses, predictions, ground_truth, heat_fig, heat_ax, epoch_counter):
        # Construct covariance matrix

        # plt.clf()
        predictions = np.squeeze(predictions)
        predictions = predictions / np.max(predictions)

        # print("Predictions:", predictions)
        # print("Ground Truths:", ground_truth)
        # heat_fig = plt.figure(figsize=(8, 6))
        # heat_ax = heat_fig.add_subplot(111, projection='3d')
        # print(np.shape(predictions))
        heat_ax.clear()
        for i in range(np.size(predictions)):
            pose = poses[i]

            x_pred = pose[0]
            y_pred = pose[1]
            r_pred = math.atan2(pose[3], pose[2])
            # print(predictions[i])

            heat_ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=2, color="red",
                         alpha=np.clip(predictions[i] - 0.1, 0, 1))  # , label='PP')

        # Add ground truth as a blue dot
        gt_x = float(ground_truth["x"])
        gt_y = float(ground_truth["y"])
        gt_z = float(ground_truth["r"])
        gt_z = math.radians(gt_z)
        if gt_z > math.pi:
            gt_z = gt_z - 2 * math.pi
        heat_ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")  # , label='GT')

        # Set axis labels and titles
        heat_ax.set_xlabel("X-coor")
        heat_ax.set_ylabel("Y-coor")
        heat_ax.set_zlabel("Z-coor")
        heat_ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
        heat_ax.legend()

        # Calculate the average distance between predicted poses and ground truth poses
        # distances = []
        # for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
        #    gt_x, gt_y, gt_z = map(float, ground_truths[i])
        #    distance = np.sqrt((x_pred - gt_x) ** 2 + (y_pred - gt_y) ** 2 + (r_pred - gt_z) ** 2)
        #    distances.append(distance)

        # average_distance = np.mean(distances)
        print("Done Plotting")
        # Create a unique filename for the plot
        filename = f"heatmap_{len(ground_truth)}.png"
        # Save the plot as a PNG image in the 'plots' folder
        # plt.show()
        heat_fig.canvas.draw()
        heat_fig.canvas.flush_events()
        # plt.savefig(
        #    f'/{filename}')
        # plt.close(fig)

        # return average_distance