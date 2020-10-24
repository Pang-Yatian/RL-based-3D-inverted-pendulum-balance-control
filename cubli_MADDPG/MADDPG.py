import tensorflow as tf
import numpy as np
from Replay_Buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, name, train, alpha=0.001, beta=0.002, gamma=0.9):
        super(Agent, self).__init__()
        self.agent_name = name
        self.max_action = 30
        self.min_action = -30
        self.action = 0
        self.train = train
        self.gamma = gamma

        self.actor_model = ActorNetwork(name=str(self.agent_name) + 'actor')
        self.target_actor_model = ActorNetwork(name=str(self.agent_name) + 'target_actor')
        self.critic_model = CriticNetwork(name='critic')
        self.target_critic_model = CriticNetwork(name='target_critic')

        self.actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

    def save_model(self):
        print("------saving model------")
        self.actor_model.save_weights(self.actor_model.checkpoint_file)

    def load_model(self):
        print("------loading model------")
        self.actor_model.load_weights(self.actor_model.checkpoint_file)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        action = self.actor_model.call(state)
        if self.train:
            action += tf.random.normal(shape=[1], mean=0.0, stddev=0.5)
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        return action[0]

    def fresh_targetnet(self):
        for a, b in zip(self.actor_model.variables, self.target_actor_model.variables):
            new_actor_variables = 0.1 * a + 0.9 * b
            b.assign(new_actor_variables)
        for c, d in zip(self.critic_model.variables, self.target_critic_model.variables):
            new_critic_variables = 0.1 * c + 0.9 * d
            d.assign(new_critic_variables)
        #print("******target net refreshed******")


class CentralController:
    def __init__(self, train, gamma=0.9, batch_size=1024, env=None, buffer_size=1000000):
        self.train = train
        self.gamma = gamma
        self.batch_size = batch_size
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.env = env
        self.memory = ReplayBuffer(buffer_size)
        self.remember_n = 0

        self.state = []
        self.next_state = []
        self.actions = []

        self.agent1 = Agent(name='agent1_', train=self.train)
        self.agent2 = Agent(name='agent2_', train=self.train)
        self.agent3 = Agent(name='agent3_', train=self.train)

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.reward = 0
        self.total_reward = 0

    def interaction(self):
        self.agent1.action = self.agent1.choose_action(self.state)
        self.agent2.action = self.agent2.choose_action(self.state)
        self.agent3.action = self.agent3.choose_action(self.state)

        self.next_state, self.reward, self.done, info = \
            self.env.step(self.agent1.action, self.agent2.action, self.agent3.action)

        self.reward = self.reward if not self.done else -1

        self.total_reward += self.reward

    def remember(self):
        self.remember_n += 1
        self.memory.store(self.state, self.agent1.action, self.agent2.action, self.agent3.action,
                          self.reward, self.next_state, self.done)

    def update_states(self):
        self.state = self.next_state

    def learn(self):
        if len(self.memory.Memory) <= self.batch_size:
            return
        #agent1
        batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)
        batch_all_action = tf.concat([batch_action1, batch_action2, batch_action3], axis=1)
        with tf.GradientTape() as tape:
            target_action1 = self.agent1.target_actor_model.call(batch_next_state)
            target_action2 = self.agent2.target_actor_model.call(batch_next_state)
            target_action3 = self.agent3.target_actor_model.call(batch_next_state)
            target_actions = tf.concat([target_action1, target_action2, target_action3], axis=1)
            target_critic_value = tf.squeeze(self.agent1.target_critic_model.call(batch_next_state, action=target_actions), 1)
            critic_value = tf.squeeze(self.agent1.critic_model.call(batch_state, batch_all_action), 1)
            target = batch_reward + self.gamma * target_critic_value * (1 - batch_done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.agent1.critic_model.variables)
        self.agent1.critic_model.optimizer.apply_gradients(zip(critic_network_gradient, self.agent1.critic_model.variables))

        with tf.GradientTape() as tape:
            new_policy_action1 = self.agent1.actor_model.call(batch_state)
            new_policy_actions = tf.concat([new_policy_action1, batch_action2, batch_action3], axis=1)
            actor_loss = -self.agent1.critic_model.call(batch_state, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.agent1.actor_model.variables)
        self.agent1.actor_model.optimizer.apply_gradients(zip(actor_network_gradient, self.agent1.actor_model.variables))
        #agent2
        batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)
        batch_all_action = tf.concat([batch_action1, batch_action2, batch_action3], axis=1)
        with tf.GradientTape() as tape:
            target_action1 = self.agent1.target_actor_model.call(batch_next_state)
            target_action2 = self.agent2.target_actor_model.call(batch_next_state)
            target_action3 = self.agent3.target_actor_model.call(batch_next_state)
            target_actions = tf.concat([target_action1, target_action2, target_action3], axis=1)
            target_critic_value = tf.squeeze(
                self.agent2.target_critic_model.call(batch_next_state, action=target_actions), 1)
            critic_value = tf.squeeze(self.agent2.critic_model.call(batch_state, batch_all_action), 1)
            target = batch_reward + self.gamma * target_critic_value * (1 - batch_done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.agent2.critic_model.variables)
        self.agent2.critic_model.optimizer.apply_gradients(
            zip(critic_network_gradient, self.agent2.critic_model.variables))

        with tf.GradientTape() as tape:
            new_policy_action2 = self.agent2.actor_model.call(batch_state)
            new_policy_actions = tf.concat([batch_action1, new_policy_action2, batch_action3], axis=1)
            actor_loss = -self.agent2.critic_model.call(batch_state, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.agent2.actor_model.variables)
        self.agent2.actor_model.optimizer.apply_gradients(
            zip(actor_network_gradient, self.agent2.actor_model.variables))
        #agent3
        batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)
        batch_all_action = tf.concat([batch_action1, batch_action2, batch_action3], axis=1)
        with tf.GradientTape() as tape:
            target_action1 = self.agent1.target_actor_model.call(batch_next_state)
            target_action2 = self.agent2.target_actor_model.call(batch_next_state)
            target_action3 = self.agent3.target_actor_model.call(batch_next_state)
            target_actions = tf.concat([target_action1, target_action2, target_action3], axis=1)
            target_critic_value = tf.squeeze(
                self.agent3.target_critic_model.call(batch_next_state, action=target_actions), 1)
            critic_value = tf.squeeze(self.agent3.critic_model.call(batch_state, batch_all_action), 1)
            target = batch_reward + self.gamma * target_critic_value * (1 - batch_done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.agent3.critic_model.variables)
        self.agent1.critic_model.optimizer.apply_gradients(
            zip(critic_network_gradient, self.agent3.critic_model.variables))

        with tf.GradientTape() as tape:
            new_policy_action3 = self.agent3.actor_model.call(batch_state)
            new_policy_actions = tf.concat([batch_action1, batch_action2, new_policy_action3], axis=1)
            actor_loss = -self.agent3.critic_model.call(batch_state, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.agent3.actor_model.variables)
        self.agent3.actor_model.optimizer.apply_gradients(
            zip(actor_network_gradient, self.agent3.actor_model.variables))

    def fresh_para(self):
        if self.remember_n % 100 == 0:
            self.agent1.fresh_targetnet()
            self.agent2.fresh_targetnet()
            self.agent3.fresh_targetnet()

    def load_model(self):
        self.agent1.load_model()
        self.agent2.load_model()
        self.agent3.load_model()