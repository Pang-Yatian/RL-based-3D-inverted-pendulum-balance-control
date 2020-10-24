import tensorflow as tf
import numpy as np
from Replay_Buffer import ReplayBuffer
from networks import EvalNetwork, TargetNetwork


class Agent:
    def __init__(self, train, learning_rate=0.0001, env=None, gamma=0.9, buffer_size=1000000, batch_size=64,
                 epsilon=0.1, epsilon_min=0.001, epsilon_decay_rate=0.999997):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.train = train

        self.eval_model = EvalNetwork(name='eval_net')
        self.target_model = TargetNetwork(name='target_net')
        self.eval_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def save_model(self):
        print("------saving model------")
        self.eval_model.save_weights(self.eval_model.checkpoint_file)

    def load_model(self):
        print("------loading model------")
        self.eval_model.load_weights(self.eval_model.checkpoint_file)

    def epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate

    def choose_action(self, state):
        if self.train:
            self.epsilon_decay()
        else:
            self.epsilon = self.epsilon_min

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, 13, 1)
        else:
            q_values = self.eval_model.call(np.expand_dims(state, axis=0))
            action = tf.argmax(q_values, axis=-1).numpy()
        action = action[0]
        return action

    def fresh_targetnet(self):
        for a, b in zip(self.eval_model.variables, self.target_model.variables):
            b.assign(a)
        print("******target net refreshed******")

    def learn(self):
        if len(self.memory.Memory) <= self.batch_size:
            return
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)
        current_q_values = self.eval_model.call(batch_next_state)
        max_actions = tf.argmax(current_q_values, axis=-1)
        target_q_values = tf.reduce_sum(self.target_model(batch_next_state) * tf.one_hot(max_actions, depth=13), axis=1)
        y_batch = batch_reward + (self.gamma * target_q_values) * (1 - batch_done)
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mean_squared_error(
                y_true=y_batch,
                y_pred=tf.reduce_sum(self.eval_model.call(batch_state) * tf.one_hot(batch_action, depth=13), axis=1)
            )
        grads = tape.gradient(loss, self.eval_model.variables)
        self.eval_model.optimizer.apply_gradients(grads_and_vars=zip(grads, self.eval_model.variables))



