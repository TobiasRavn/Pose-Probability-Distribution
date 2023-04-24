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
def compute_loss(mlp_model, vision_description, gt, training=True):
    logits = mlp_model([vision_description, gt],
                                 training=training)[Ellipsis, 0]

    logits_norm = tf.nn.softmax(logits, axis=-1)
    #                                                            area              samples
    loss_value = -tf.reduce_mean(tf.math.log(logits_norm[:, -1]/(((0.6**2)*3.1415*2)/gt.shape[1]))) #index -1 because last one is the correct pose
    return loss_value

@tf.function
def train_step(vision_model, mlp_model, optimizer, images, gts):
    with tf.GradientTape() as tape:
        vision_description = vision_model(images, training=True)
        loss = compute_loss(mlp_model, vision_description, gts)
    grads = tape.gradient(
        loss,
        vision_model.trainable_variables + mlp_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, vision_model.trainable_variables +
            mlp_model.trainable_variables))
    return loss


def validation_step(vision_model, mlp_model, optimizer, images, gts):
    vision_description = vision_model(images, training=False)
    loss = compute_loss(mlp_model, vision_description, gts, training=False)

    return loss

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


def get_random_poses_plus_correct(position_samples, ground_truth):
    poses = np.zeros(position_samples, 4)

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
batches = [train_data[x:x+batch_size] for x in range(0, len(train_data), batch_size)]
for epoch in range(epochs):
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

    validations_set=random.sample(validations_set, 100)

    batches = [validations_set[x:x + batch_size] for x in range(0, len(validations_set), batch_size)]
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
        loss = train_step(descriptor.vision_model, mlp_model, optimizer, images, ground_truths)
        print("loss: ", loss.numpy(), " time: ", time.time() - st)
    validation_loss_list.append(loss)
