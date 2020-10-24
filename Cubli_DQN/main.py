import gym
import pybullet as p
import numpy as np
from NatureDQN import Agent
import cubli_robot
import tensorflow as tf
from collections import deque

train = False
Max_episodes = 50000
every_episode_fresh = 50
best_avg_reward = 0


if train:
    p.connect(p.DIRECT)
else:
    p.connect(p.GUI)

if __name__ == '__main__':
    env = gym.make("cublirobot-v0")
    agent = Agent(train=train, env=env)
    total_reward_list = deque(maxlen=50)
    writer = tf.summary.create_file_writer("./mylogs")

    if not train:
        agent.load_model()
        state = env.reset()
        done = False
        roll = -180
        speed = 0.1
        while True:     #not done:
            if (roll > 0 and roll < 360):
                p.resetDebugVisualizerCamera(0.7, 90+roll, -30, [0, 0, 0.1])
            elif roll <= 0:
                p.resetDebugVisualizerCamera(0.7, 90, -30, [0, 0, 0.1])
            else:
                p.resetDebugVisualizerCamera(0.7, 90, -30, [0, 0, 0.1])
            roll += speed
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
    else:
        with writer.as_default():
            for episode in range(Max_episodes):
                state = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    reward = reward if not done else -0.5
                    total_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.learn()
                    state = next_state
                total_reward_list.append(total_reward)
                tf.summary.scalar("reward", total_reward, step=episode)
                writer.flush()
                if episode % every_episode_fresh == 0:
                    agent.fresh_targetnet()
                if np.mean(total_reward_list) > best_avg_reward:
                    agent.save_model()
                    best_avg_reward = np.mean(total_reward_list)
                print("episode:", episode, "score", total_reward, "epsilon", agent.epsilon)
