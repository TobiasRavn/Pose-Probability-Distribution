import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#https://keras.io/guides/writing_a_training_loop_from_scratch/
class MLP:
    def __init__(self, lenDiscriptors, lenPose):
        self.lenDiscriptors = lenDiscriptors
        self.lenPose = lenPose
        self.learning_rate = 1e-2
        #define MLP
        inputs = keras.Input(shape=(self.lenDiscriptors+self.lenPose,), name="Descrip_and_pose")
        x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(256, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(1, name="predictions")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        #define optimizer and loss function
        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        #define metrics used to track accuracy for training and validation
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    def get(self, features, position):
        #pass features and position through MLP
        x = self.convert_to_tensor(features, position)
        logits = self.model(x, training=False)
        return logits.numpy().transpose()[0]

    def convert_to_tensor(self, features, position):
        # convert shape from (n,) to (None, n)
        position = np.expand_dims(position, axis=0)
        features = np.expand_dims(features, axis=0)

        x = np.concatenate((features, position), axis=1)
        x = tf.convert_to_tensor(x)
        return x

    def train_single(self, features, position, output):
        #train MLP
        #concatenate features and position
        x = self.convert_to_tensor(features, position)
        #y = tf.convert_to_tensor(output)
        #convert output to float32 tensor and reshape
        y = tf.convert_to_tensor(output, dtype=tf.float32)
        y = tf.reshape(y, (1,1))
        # train step
        loss_value = self.train_step(x, y)
        print("x shape: ", x.shape, " y shape: ", y.shape)
        #print("Training loss: ", loss_value.numpy())
        return loss_value.numpy()
    
    def train(self, features, position, output):
        #train MLP
        #copy the 1x2048 feature matrix into 101x2048 matrix
        features = np.repeat(features, 101, axis=0)
        #make tensor of size (None, lenDiscriptors+lenPose)
        x = np.concatenate((features, position), axis=1)
        #convert to tensor
        x = tf.convert_to_tensor(x)
        #y = tf.convert_to_tensor(output)
        #convert output to float32 tensor and reshape
        y = tf.convert_to_tensor(output, dtype=tf.float32)
        # train step
        loss_value , logits = self.train_step(x, y)
        # print("logits: ", logits)
        logits_norm = tf.nn.softmax(logits, axis=0)
        print("loss value: ", loss_value)
        print("should be 0 " , np.mean(logits_norm.numpy()[0:99]), " should be 1 ", np.mean(logits_norm.numpy()[-1]))
        #print("loss value: ", loss_value)
        #print("Training loss: ", loss_value.numpy())
        return loss_value.numpy()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            logits_norm = tf.nn.softmax(logits, axis=0)
            loss_value = -tf.math.log(logits_norm[-1]*10)
            #loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value , logits

    def save_MLP(self, path):
        pass
        #save MLP
        pass
    def load_MLP(self, path):
        pass
        #load MLP
        pass