import tensorflow as tf
import os


class EvalNetwork(tf.keras.Model):
    def __init__(self, name='eval_net', checkpoint_dir='tmp/ddqn'):
        super().__init__()
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddqn')

        self.Dense1 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense2 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense3 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense4 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense5 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense6 = tf.keras.layers.Dense(units=13)

    def call(self, inputs):
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        outputs = self.Dense6(x)
        return outputs


class TargetNetwork(tf.keras.Model):
    def __init__(self, name='target_net', checkpoint_dir='tmp/ddqn'):
        super().__init__()
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddqn')

        self.Dense1 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense2 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense3 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense4 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense5 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense6 = tf.keras.layers.Dense(units=13)

    def call(self, inputs):
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        outputs = self.Dense6(x)
        return outputs
