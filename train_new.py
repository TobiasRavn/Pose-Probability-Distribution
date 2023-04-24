import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import time
#own libs
from lib.load_image import *
from lib.Pose_Accumulator import *
from lib.Descriptor import *

@tf.function
def compute_loss(mlp_model, vision_description, poses, training=True):
    logits = mlp_model([vision_description, poses],
                                 training=training)[Ellipsis, 0]

    logits_norm = tf.nn.softmax(logits, axis=-1)
    #                                                            area              samples
    loss_value = -tf.reduce_mean(tf.math.log(logits_norm[:, -1]/(((0.6**2)*3.1415*2)/poses.shape[1]))) #index -1 because last one is the correct pose
    return loss_value

@tf.function
def train_step(vision_model, mlp_model, optimizer, images, poses):
    with tf.GradientTape() as tape:
        vision_description = vision_model(images, training=True)
        loss = compute_loss(mlp_model, vision_description, poses)
    grads = tape.gradient(
        loss,
        vision_model.trainable_variables + mlp_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, vision_model.trainable_variables +
            mlp_model.trainable_variables))
    return loss


def validation_step(vision_model, mlp_model, images, poses):
    vision_description = vision_model(images, training=True)
    loss = compute_loss(mlp_model, vision_description, poses, training=False)

    return loss

def generate_pdf(vision_model, mlp_model, images, poses):


    vision_description = vision_model(images, training=True)

    logits = mlp_model([vision_description, poses],
                       training=False)[Ellipsis, 0]

    logits_norm = tf.nn.softmax(logits, axis=-1)

    return logits_norm


def plotHeatmap(poses, predictions, ground_truth):
    # Construct covariance matrix

    predictions=np.squeeze(predictions)
    predictions=predictions/np.max(predictions)

    print("Predictions:", predictions)
    print("Ground Truths:", ground_truth)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    print(np.shape(predictions))
    for i in range(np.size(predictions)):

        pose=poses[i]

        x_pred=pose[0]
        y_pred=pose[1]
        r_pred=math.atan2(pose[3],pose[2])
        #print(predictions[i])
        ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=10, color="red",alpha=predictions[i]-0.1)  # , label='PP')

        # Add ground truth as a blue dot
        gt_x = float(ground_truth["x"])
        gt_y = float(ground_truth["y"])
        gt_z = float(ground_truth["r"])
        gt_z=math.radians(gt_z)
        ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")  # , label='GT')

    # Set axis labels and titles
    ax.set_xlabel("X-coor")
    ax.set_ylabel("Y-coor")
    ax.set_zlabel("Z-coor")
    ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
    ax.legend()

    # Calculate the average distance between predicted poses and ground truth poses
    #distances = []
    #for i, (x_pred, y_pred, r_pred) in enumerate(predictions):
    #    gt_x, gt_y, gt_z = map(float, ground_truths[i])
    #    distance = np.sqrt((x_pred - gt_x) ** 2 + (y_pred - gt_y) ** 2 + (r_pred - gt_z) ** 2)
    #    distances.append(distance)

    #average_distance = np.mean(distances)
    print("Done Plotting")
    # Create a unique filename for the plot
    filename = f"heatmap_{len(ground_truth)}.png"
    # Save the plot as a PNG image in the 'plots' folder
    plt.show()
    #plt.savefig(
    #    f'/{filename}')
    #plt.close(fig)

    #return average_distance

# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')

#gather all files names
dir = "blenderproc/data_500_first"


dir = "blenderproc/data"
files=glob.glob(dir+"/*.hdf5")

random.shuffle(files)
#split files into training and validation data
train_data = files[:int(len(files)*0.8)]
vali_data = files[int(len(files)*0.8):]


#load first image to use img size
image, ground_truth = load_image(files[0])
img=np.array(image)
imgSize=img.shape
#init descriptor
descriptor=Descriptor(imgSize)
#init mlp model
lenDiscriptors = 2048
lenPose = 4
learning_rate = 1e-4

#define MLP from IPDF
input_visual = tfkl.Input(shape=(lenDiscriptors,))
visual_embedding = tfkl.Dense(256)(input_visual)
input_query = tfkl.Input(shape=(None, lenPose,))
query_embedding = tfkl.Dense(256)(input_query)
output = visual_embedding[:, tf.newaxis] + query_embedding
output = tfkl.ReLU()(output)
output = tfkl.Dense(256, 'relu')(output)
output = tfkl.Dense(1)(output)
mlp_model = tf.keras.models.Model(
    inputs=[input_visual, input_query],
    outputs=output)

#define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#init plot of loss using plt
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.show()

x_min, x_max = -0.3, 0.3
y_min, y_max = -0.3, 0.3
r_min, r_max = 0, 360

position_samples = 100
loss_list = []
validation_loss_list=[]
epochs=10
batch_size=4 
#split files into batches of 10

def get_all_poses(step_x, step_y, step_r):
    x_num = round((x_max - x_min + step_x) / step_x)
    y_num = round((y_max - y_min + step_y) / step_y)
    r_num = round((360) / step_r)
    x_range = np.linspace(x_min, x_max, int(x_num))
    y_range = np.linspace(y_min, x_max, int(y_num))
    r_range = np.linspace(0, 360 - step_r, int(r_num))

    size = x_num * y_num * r_num
    allPoses = np.zeros([size, 4])
    count=0

    for x in x_range:
        for y in y_range:
            for r in r_range:
                r_rad = math.radians(r)

                allPoses[count] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
                count += 1


    return allPoses
def get_random_poses_plus_correct(position_samples, ground_truth):
    poses = np.zeros((position_samples, 4))

    x = ground_truth["x"]
    y = ground_truth["y"]
    r = ground_truth["r"]
    x, y, r = float(x), float(y), float(r)
    r_rad = math.radians(r)
    poses[-1] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
    for j in range(position_samples - 1):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        r = random.uniform(r_min, r_max)
        r_rad = math.radians(r)
        poses[j] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
    return poses

#batches = [files[x:x+batch_size] for x in range(0, len(files), batch_size)]

for epoch in range(epochs):

    file = random.sample(vali_data, 1)

    image, ground_truth = load_image(file[0])
    images=[image]
    images = tf.convert_to_tensor(images)
    all_poses = get_all_poses(0.05, 0.05, 10)
    predictions = generate_pdf(descriptor.vision_model, mlp_model, images, all_poses)
    plotHeatmap(all_poses, predictions, ground_truth)



    random.shuffle(train_data)
    batches = [train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)]
    for count, batch in enumerate(batches):
        #dont use batch size here because the last one might be smaller
        #init images 
        images = np.zeros((len(batch), imgSize[0], imgSize[1], imgSize[2]))
        #init ground truth
        ground_truths = np.zeros((len(batch),position_samples, 4))
        for i, file in enumerate(batch):
            image, ground_truth = load_image(file)
            images[i] = image

            ground_truths[i]=get_random_poses_plus_correct(position_samples,ground_truth)
        
        #convert numpy arrays to tensors
        images = tf.convert_to_tensor(images)
        ground_truths = tf.convert_to_tensor(ground_truths)

        #time to train_step
        st = time.time()
        loss = train_step(descriptor.vision_model, mlp_model, optimizer, images, ground_truths)
        print("loss: ", loss.numpy()," time: ", time.time()-st)
        loss_list.append(loss.numpy())
        ax.clear()
        ax.plot(loss_list)
        fig.canvas.draw()
        fig.canvas.flush_events()



    #Validation

    validations_set=random.sample(vali_data, 50)

    batches = [validations_set[x:x + batch_size] for x in range(0, len(validations_set), batch_size)]

    validation_loses=[]

    for count, batch in enumerate(batches):
        images = np.zeros((len(batch), imgSize[0], imgSize[1], imgSize[2]))
        # init ground truth
        ground_truths = np.zeros((len(batch), position_samples, 4))
        for i, file in enumerate(batch):
            image, ground_truth = load_image(file)
            images[i] = image

            ground_truths[i] = get_random_poses_plus_correct(position_samples,ground_truth)

        # convert numpy arrays to tensors
        images = tf.convert_to_tensor(images)
        ground_truths = tf.convert_to_tensor(ground_truths)

        # time to train_step
        st = time.time()
        loss = validation_step(descriptor.vision_model, mlp_model, optimizer, images, ground_truths)
        print("loss: ", loss.numpy(), " time: ", time.time() - st)
        validation_loses.append(loss)






    validation_loss=np.mean(np.array(validation_loses))
    validation_loss_list.append(validation_loss)