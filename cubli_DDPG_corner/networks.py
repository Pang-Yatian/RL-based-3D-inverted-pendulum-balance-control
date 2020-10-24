import tensorflow as tf
import os


class CriticNetwork(tf.keras.Model):
    def __init__(self, name='critic', checkpoint_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg')

        self.Dense1 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense2 = tf.keras.layers.Dense(units=72, activation=tf.nn.relu)
        self.Dense3 = tf.keras.layers.Dense(units=108, activation=tf.nn.relu)
        self.Dense4 = tf.keras.layers.Dense(units=72, activation=tf.nn.relu)
        self.Dense5 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense6 = tf.keras.layers.Dense(units=1)

    def call(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        outputs = self.Dense6(x)
        return outputs


class ActorNetwork(tf.keras.Model):
    def __init__(self, name='actor', checkpoint_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg')

        self.Dense1 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense2 = tf.keras.layers.Dense(units=72, activation=tf.nn.relu)
        self.Dense3 = tf.keras.layers.Dense(units=108, activation=tf.nn.relu)
        self.Dense4 = tf.keras.layers.Dense(units=72, activation=tf.nn.relu)
        self.Dense5 = tf.keras.layers.Dense(units=36, activation=tf.nn.relu)
        self.Dense6 = tf.keras.layers.Dense(units=3, activation=tf.nn.tanh)

    def call(self, state):
        inputs = state
        x = self.Dense1(inputs)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        outputs = self.Dense6(x)
        return outputs
