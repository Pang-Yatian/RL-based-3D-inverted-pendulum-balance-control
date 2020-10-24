import tensorflow as tf
import numpy as np
from Replay_Buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, train, alpha=0.001, beta=0.002, env=None, gamma=0.9, buffer_size=1000000, batch_size=512):
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.train = train

        self.actor_model = ActorNetwork(name='actor')
        self.target_actor_model = ActorNetwork(name='target_actor')
        self.critic_model = CriticNetwork(name='critic')
        self.target_critic_model = CriticNetwork(name='target_critic')

        self.actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def save_model(self):
        print("------saving model------")
        self.actor_model.save_weights(self.actor_model.checkpoint_file)
        self.critic_model.save_weights(self.critic_model.checkpoint_file)

    def load_model(self):
        print("------loading model------")
        self.actor_model.load_weights(self.actor_model.checkpoint_file)
        self.critic_model.load_weights(self.critic_model.checkpoint_file)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        action = self.actor_model.call(state)
        if self.train:
            action += tf.random.normal(shape=[1], mean=0.0, stddev=1)
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        return action[0]

    def fresh_targetnet(self):
        for a, b in zip(self.actor_model.variables, self.target_actor_model.variables):
            new_actor_variables = 0.99 * b + 0.01 * a
            b.assign(new_actor_variables)
        for c, d in zip(self.critic_model.variables, self.target_critic_model.variables):
            new_critic_variables = 0.99 * d + 0.01 * c
            d.assign(new_critic_variables)
        #print("******target net refreshed******")

    def learn(self):
        if len(self.memory.Memory) <= self.batch_size:
            return
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)

        with tf.GradientTape() as tape:
            target_action = self.target_actor_model.call(batch_next_state)
            target_critic_value = tf.squeeze(self.target_critic_model.call(batch_next_state, action=target_action), 1)
            critic_value = tf.squeeze(self.critic_model.call(batch_state, batch_action), 1)
            target = batch_reward + self.gamma * target_critic_value * (1 - batch_done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic_model.variables)
        self.critic_model.optimizer.apply_gradients(zip(critic_network_gradient, self.critic_model.variables))

        with tf.GradientTape() as tape:
            new_policy_action = self.actor_model.call(batch_state)
            actor_loss = -self.critic_model.call(batch_state, new_policy_action)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor_model.variables)
        self.actor_model.optimizer.apply_gradients(zip(actor_network_gradient, self.actor_model.variables))

        self.fresh_targetnet()
