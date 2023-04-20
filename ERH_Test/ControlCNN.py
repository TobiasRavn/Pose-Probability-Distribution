import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image

# ... (include all other necessary imports and definitions) ...
import torch.nn as nn
import torch.nn.functional as F

import h5py
import ast
import glob

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseEstimationDataset(Dataset):
    def __init__(self, image_arrays, ground_truths, transform=None):
        self.image_arrays = image_arrays
        self.ground_truths = ground_truths
        self.transform = transform

    def __len__(self):
        return len(self.image_arrays)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_arrays[idx])
        if self.transform:
            image = self.transform(image)
        ground_truth = [float(x) for x in self.ground_truths[idx]]
        print("Ground truth:", ground_truth)
        return image, torch.tensor(ground_truth, dtype=torch.float32)
        

class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plotHeatmap(predictions, ground_truths):
    # Construct covariance matrix
    var_x = 0.1
    var_y = 0.1
    var_z = 0.1
    corr_xy = 0.5
    cov_matrix = np.array([[var_x, corr_xy*np.sqrt(var_x*var_y), 0],
                           [corr_xy*np.sqrt(var_x*var_y), var_y, 0],
                           [0, 0, var_z]])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
        # Calculate probabilities using multivariate Gaussian distribution
        xyz = np.vstack((x_pred, y_pred, r_pred)).T
        prob = np.exp(-0.5 * np.sum(xyz.dot(np.linalg.inv(cov_matrix)) * xyz, axis=1))

        # Create a scatter plot with a colorbar
        cmap = plt.get_cmap("hsv")
        norm = plt.Normalize(vmin=0, vmax=1)
        colors = cmap(norm(prob))

        sc = ax.scatter(x_pred, y_pred, r_pred, c=colors, cmap=cmap, s=20, edgecolors="black")

        # Add predicted pose as a red dot and ground truth as a blue dot
        ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=10, color="red")
        ax.plot([ground_truths[i][0]], [ground_truths[i][1]], [ground_truths[i][2]], marker='o', markersize=10, color="blue")

    # Move colorbar to the right side
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, shrink=0.7, aspect=10, 
                        fraction=0.15, label='Probability')
    cbar.ax.tick_params(labelsize=10)
    cax = cbar.ax
    cax.yaxis.set_label_position('right')
    cax.yaxis.set_ticks_position('right')
    cax.set_ylabel('Probability', rotation=-90, va='bottom', fontsize=14, labelpad=10)

    # Set axis labels and titles
    ax.set_xlabel("X-coor")
    ax.set_ylabel("Y-coor")
    ax.set_zlabel("Z-coor")
    ax.set_title("Probability of Pose Estimation of Cup in Three Planes", fontsize=20)

    plt.show()


def predict_poses(model, image_arrays, transform):
    model.eval()
    with torch.no_grad():
        outputs = []
        for image_array in image_arrays:
            image = Image.fromarray(image_array)
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            x, y, r = output[0].cpu().numpy()
            outputs.append((x, y, r))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir = "/Users/reventlov/Documents/AI_Projects/PAR_AI_PoseE/data"
    files = glob.glob(dir + "/*.hdf5")
    

    image_arrays = []
    validate_image_arrays = []
    ground_truths = []
    validate_ground_truths = []


    for i, file in enumerate(files):
        image, ground_truth = load_image(file)
        
        if i % 100 == 0:
            validate_image_arrays.append(image)
            validate_ground_truths.append(ground_truth)
        else:
            image_arrays.append(image)
            ground_truths.append(ground_truth)

    print(f"Number of training images: {len(image_arrays)}")
    print(f"Number of validation images: {len(validate_image_arrays)}")
    input("Press any key to continue...")
        

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = PoseEstimationDataset(image_arrays, ground_truths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PoseEstimationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, ground_truth in dataloader:
            images, ground_truth = images.to(device), ground_truth.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

    torch.save(model.state_dict(), "pose_estimation_model.pth")

    # Replace the for loop for validation images with the following code:
    predictions = predict_poses(model, validate_image_arrays, transform)
    validate_ground_truths_list = [list(map(float, gt)) for gt in validate_ground_truths]

    # Compute errors
    errors = []
    for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
        validate_ground_truth = validate_ground_truths_list[i]
        error_x = abs(x_pred - validate_ground_truth[0])
        error_y = abs(y_pred - validate_ground_truth[1])
        error_r = abs(r_pred - validate_ground_truth[2])
        errors.append((error_x, error_y, error_r))
        print(f'Error for validation image {i}: x = {error_x}, y = {error_y}, r = {error_r}')

    avg_error = [sum(errors[i])/len(errors) for i in range(3)]
    print(f'Average error for all validation images: x = {avg_error[0]}, y = {avg_error[1]}, r = {avg_error[2]}')

    
    plotHeatmap(predictions, validate_ground_truths_list)
    

if __name__ == "__main__":
    main()
