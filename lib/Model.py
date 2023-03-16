#define class IPDF
from MLP import *
from Descriptor import *
from Pose_Accumulator import *

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