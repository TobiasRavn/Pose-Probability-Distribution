#define class IPDF

class IPDF:
    def __init__(self):
        self.x_min, self.x_max = -0.3, 0.3
        self.y_min, self.y_max = -0.3, 0.3
        self.r_min, self.r_max = 0, 360
        #init resnet

        #init MLP

    def image_descriptor(self, image):
        #extract features from image resnet18

        #return features

    def MLP(self, features, position):
        #pass features and position through MLP

        #return prediction

    def MLP_train(self, features, position, output):
        #train MLP

    def sample_space(self, x_step, y_step, r_step):
        #sample space

        #return matrix of size [(x_max-x_min)/x_step), (y_max-y_min)/y_step),(r_max-r_min)/r_step)]

    def sample_space_train(self, image, ground_truth, x_step, y_step, r_step):
        #sample space

    def save_MLP(self, path):
        #save MLP

    def load_MLP(self, path):
        #load MLP

    def plotHeatmap(self, pdf_matrix):
        #plot heatmap