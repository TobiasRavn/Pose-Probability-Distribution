#define class IPDF
from lib.MLP import *
from lib.Descriptor import *
from lib.Pose_Accumulator import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Controller:
    def __init__(self, imgSize):
        self.x_min, self.x_max = -0.3, 0.3
        self.y_min, self.y_max = -0.3, 0.3
        self.r_min, self.r_max = 0, 360
        #init resnet

        self.mlp=MLP(50,4)
        self.descriptor=Descriptor(imgSize)
        #init MLP

    def sample_space(self, image, x_step, y_step, r_step, truth=0,training=False):
        descriptor = self.descriptor.get_image_descriptor_array(image)
        poses=0
        if(training==True):
            poses = Pose_Accumulator(x_step, self.x_min, self.x_max, y_step, self.y_min, self.y_max, r_step,training=training, truth=truth);
        else:
            poses = Pose_Accumulator(x_step, self.x_min, self.x_max, y_step, self.y_min, self.y_max, r_step,
                                     training=training, truth=truth);

        for pose in poses:
            if(training):
                poses.result(self.mlp.train(descriptor, pose.pose(),pose.output()))
            else:
                poses.result(self.mlp.get(descriptor, pose.pose()))
        # sample space
        # return matrix of size [(x_max-x_min)/x_step), (y_max-y_min)/y_step),(r_max-r_min)/r_step)]


    def sample_space_train(self, image, ground_truth, x_step, y_step, r_step):
        #sample space
        pass


    def plotHeatmap(self, pdf_matrix):
        #plot heatmap
        # Generate random data for position and orientation
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        theta = np.random.uniform(low=0, high=360, size=1000)

        # Construct covariance matrix
        var_x = 0.1
        var_y = 0.1
        var_theta = 0.1
        corr_xy = 0.5
        cov_matrix = np.array([[var_x, corr_xy*np.sqrt(var_x*var_y), 0],
                            [corr_xy*np.sqrt(var_x*var_y), var_y, 0],
                            [0, 0, var_theta]])

        # Calculate probabilities using multivariate Gaussian distribution
        # Here, we assume that the mean is zero for simplicity
        xyz = np.vstack((x,y,theta)).T
        prob = np.exp(-0.5 * np.sum(xyz.dot(np.linalg.inv(cov_matrix)) * xyz, axis=1))

        # Map the rotation values to a color scale
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=360)
        colors = cmap(norm(theta))

        # Create a scatter plot with a colorbar
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(x, y, c=colors, cmap=cmap, s=20, edgecolors="black")
        sc.set_clim(0, 360)

        # Move colorbar to the right side
        cbar = plt.colorbar(sc, ax=ax, ticks=np.linspace(0, 360, num=9), 
                            orientation='vertical', pad=0.05, shrink=0.7, aspect=10, 
                            fraction=0.15, label='Rotation (degrees)')
        cbar.ax.tick_params(labelsize=10)
        cax = cbar.ax
        cax.yaxis.set_label_position('right')
        cax.yaxis.set_ticks_position('right')
        cax.set_ylabel('Rotation (degrees)', rotation=-90, va='bottom', fontsize=14, labelpad=10)

        # Set axis labels and titles
        ax.set_xlabel("X-coor")
        ax.set_ylabel("Y-coor")
        ax.set_title("Probability of Pose Estimation of Cup in One Plane", fontsize=20)

        plt.show()