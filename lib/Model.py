#define class IPDF
from MLP import *
from Descriptor import *
from Pose_Accumulator import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.x_min, self.x_max = -0.3, 0.3
        self.y_min, self.y_max = -0.3, 0.3
        self.r_min, self.r_max = 0, 360
        #init resnet

        self.mlp=MPL()
        self.descriptor=Descriptor()
        #init MLP





    def sample_space(self, image, x_step, y_step, r_step):

        descriptor=self.descriptor.get_image_descriptor(image)

        poses=self.__get_all_poses(x_step,y_step,r_step)
        poses=Pose_Accumulator(x_step,self.x_min,self.x_max,y_step,self.y_min,self.y_max,r_step);


        for pose in poses:
            poses.result(self.mlp.get(descriptor,pose))
        #sample space

        #return matrix of size [(x_max-x_min)/x_step), (y_max-y_min)/y_step),(r_max-r_min)/r_step)]

    def sample_space_train(self, image, ground_truth, x_step, y_step, r_step):
        #sample space



    def plotHeatmap(self, pdf_matrix):
        #plot heatmap
        # Generate some random data
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        theta = np.random.uniform(low=0, high=360, size=1000)
        prob = np.exp(-(x**2 + y**2 + (theta-180)**2/360))

        # Map the rotation values to a color scale
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(0, 360)
        colors = cmap(norm(theta))

        # Create a scatter plot with a colorbar
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(x, y, c=colors, cmap=cmap, s=20, edgecolors="black")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Rotation (degrees)")

        # Set axis labels and titles
        ax.set_xlabel("X-coor")
        ax.set_ylabel("Y-coor")
        ax.set_title("Probability of Pose Estimation of Cup in One Plane", fontsize=20)

        plt.show()