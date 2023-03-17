import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class MLP:
    def __init__(self, lenDiscriptors, lenPose):
        self.lenDiscriptors = lenDiscriptors
        self.lenPose = lenPose
        self.learning_rate = 0.001
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
        #convert features and position to tensor
        x = np.concatenate((features, position), axis=1)
        x = tf.convert_to_tensor(x)
        return x

    def train(self, features, position, output):
        #train MLP
        #concatenate features and position
        x = self.convert_to_tensor(self, features, position)
        y = tf.convert_to_tensor(output)
        # train step
        loss_value = self.train_step(x, y)
        return loss_value


    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    def save_MLP(self, path):
        #save MLP
        pass
    def load_MLP(self, path):
        #load MLP
        pass