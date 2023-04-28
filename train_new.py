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

#start timer
start_time = time.time()


@tf.function
def compute_loss(mlp_model, vision_description, poses, training=True):
    logits = mlp_model([vision_description, poses],
                                 training=training)[Ellipsis, 0]

    logits_norm = tf.nn.softmax(logits, axis=-1)
    #                                                            area/span             samples
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
    vision_description = vision_model(images, training=False)
    loss = compute_loss(mlp_model, vision_description, poses, training=False)

    return loss

def generate_pdf(vision_model, mlp_model, images, poses):


    vision_description = vision_model(images, training=True)

    logits = mlp_model([vision_description, poses],
                       training=False)[Ellipsis, 0]

    logits_norm = tf.nn.softmax(logits, axis=-1)

    return logits_norm


def plotHeatmap(poses, predictions, ground_truth, heat_fig, heat_ax, epoch_counter):
    # Construct covariance matrix

    #plt.clf()
    predictions=np.squeeze(predictions)
    predictions=predictions/np.max(predictions)

    #print("Predictions:", predictions)
    #print("Ground Truths:", ground_truth)
    #heat_fig = plt.figure(figsize=(8, 6))
    #heat_ax = heat_fig.add_subplot(111, projection='3d')
    #print(np.shape(predictions))
    heat_ax.clear()
    for i in range(np.size(predictions)):

        pose=poses[i]

        x_pred=pose[0]
        y_pred=pose[1]
        r_pred=math.atan2(pose[3],pose[2])
        #print(predictions[i])

        heat_ax.plot([x_pred], [y_pred], [r_pred], marker='o', markersize=2, color="red",alpha=np.clip(predictions[i]-0.1,0,1))  # , label='PP')

    # Add ground truth as a blue dot
    gt_x = float(ground_truth["x"])
    gt_y = float(ground_truth["y"])
    gt_z = float(ground_truth["r"])
    gt_z=math.radians(gt_z)
    if gt_z>math.pi:
        gt_z=gt_z-2*math.pi
    heat_ax.plot([gt_x], [gt_y], [gt_z], marker='o', markersize=10, color="blue")  # , label='GT')


    # Set axis labels and titles
    heat_ax.set_xlabel("X-coor")
    heat_ax.set_ylabel("Y-coor")
    heat_ax.set_zlabel("Z-coor")
    heat_ax.set_title("Predicted and Ground Truth Poses", fontsize=20)
    heat_ax.legend()

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
    #plt.show()
    heat_fig.canvas.draw()
    heat_fig.canvas.flush_events()
    #plt.savefig(
    #    f'/{filename}')
    #plt.close(fig)

    #return average_distance

# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')

#gather all files names
dir = "blenderproc/data_500_first"

#dir = "blenderproc/data"
#dir = "blenderproc/data_triangle"
#dir = "blenderproc/data_1000"
files=glob.glob(dir+"/*.hdf5")

random.shuffle(files)
#split files into training and validation data
train_data = files[:int(len(files)*0.8)]
vali_data = files[int(len(files)*0.8):]

#Set file name
current_test_name = "batch_4_epoch_100_1e_4_triangle"

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


fig_loss_epoch = plt.figure()
ax_loss_epoch = fig_loss_epoch.add_subplot(111)

x_min, x_max = -0.3, 0.3
y_min, y_max = -0.3, 0.3
r_min, r_max = 0, 360

position_samples = 100
loss_list = []
validation_loss_list=[]
epochs = 100
batch_size=4
#split files into batches of 10

def get_all_poses(x_num, y_num, r_num):

    x_min=-1
    x_max=1
    y_min=-1
    y_max=1

    step_r=360/r_num


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

    x=x/0.3
    y=y/0.3
    r_rad = math.radians(r)
    poses[-1] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])


    for j in range(position_samples - 1):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        r = random.uniform(r_min, r_max)
        r_rad = math.radians(r)
        poses[j] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
    return poses

#batches = [files[x:x+batch_size] for x in range(0, len(files), batch_size)]

epoch_counter = 0
heat_fig = plt.figure(figsize=(8, 6))
heat_ax = heat_fig.add_subplot(111, projection='3d')

epoch_lose_list = []
for epoch in range(epochs):



    temp_epoch_loss = []


#    file = random.sample(vali_data, 1)
#
#    image, ground_truth = load_image(file[0])
#    images=[image]
#    images = tf.convert_to_tensor(images)
#    all_poses = get_all_poses(0.05, 0.05, 10)
#    predictions = generate_pdf(descriptor.vision_model, mlp_model, images, all_poses)
#    plotHeatmap(all_poses, predictions, ground_truth, heat_fig,heat_ax, epoch_counter)



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
        temp_epoch_loss.append(loss.numpy())
        ax.clear()
        ax.plot(loss_list)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per Batch", fontsize=20)
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

        loss = validation_step(descriptor.vision_model, mlp_model, images, ground_truths)
        #print("loss: ", loss.numpy(), " time: ", time.time() - st)

        validation_loses.append(loss)
    
    file = random.sample(vali_data, 1)

    image, ground_truth = load_image(file[0])
    images=[image]
    images = tf.convert_to_tensor(images)
    all_poses = get_all_poses(0.05, 0.05, 10)
    predictions = generate_pdf(descriptor.vision_model, mlp_model, images, all_poses)
    plotHeatmap(all_poses, predictions, ground_truth, heat_fig,heat_ax, epoch_counter)

    print("Epoch loss: ", np.mean(np.array(temp_epoch_loss)))
    epoch_lose_list.append(np.mean(np.array(temp_epoch_loss)))
    ax_loss_epoch.clear()
    ax_loss_epoch.plot(epoch_lose_list)
    ax_loss_epoch.plot(validation_loss_list,'bo')
    ax_loss_epoch.set_xlabel("Epoch")
    ax_loss_epoch.set_ylabel("Loss")
    ax_loss_epoch.set_title("Loss over Epoch", fontsize=20)
    fig_loss_epoch.canvas.draw()
    fig_loss_epoch.canvas.flush_events()

    mlp_model.save("mlp_"+current_test_name+"_"+str(epoch_counter))
    descriptor.vision_model.save("vm_"+current_test_name+"_"+str(epoch_counter))
    heat_fig.savefig("new_training_output/Heat_map_"+current_test_name+"_"+str(epoch_counter)+".png")
    fig.savefig("new_training_output/loss_"+current_test_name+"_"+str(epoch_counter)+".png")
    epoch_counter+=1
    validation_loss=np.mean(np.array(validation_loses))
    print("Validation loss: ", validation_loss)
    validation_loss_list.append(validation_loss)
    end_time = time.time()

#When the trainign started as a date and time
#print("Traning started at: ", start_time.strftime("%d/%m/%Y %H:%M:%S")," and ended at: ", end_time.strftime("%d/%m/%Y %H:%M:%S"))
print("Training took: ", end_time-start_time, " seconds")


