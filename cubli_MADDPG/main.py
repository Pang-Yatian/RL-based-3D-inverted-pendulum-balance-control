import gym
import pybullet as p
import numpy as np
from MADDPG import CentralController
import tensorflow as tf
from collections import deque
import cubli_robot


train = False
Max_episodes = 50000

best_avg_reward = 0

if train:
    p.connect(p.DIRECT)
else:
    p.connect(p.GUI)

if __name__ == '__main__':
    env = gym.make("cublirobot-v0")
    CC = CentralController(train, env=env)
    total_reward_list = deque(maxlen=50)
    writer = tf.summary.create_file_writer("./mylogs")

    if not train:
        roll = -180
        speed = 0.3
        CC.load_model()
        CC.reset()
        while True:     #not CC.done:
            if (roll > 0 and roll < 360):
                p.resetDebugVisualizerCamera(0.7, roll, -30, [0, 0, 0.1])
            elif roll<= 0:
                p.resetDebugVisualizerCamera(0.7, 0, -30, [0, 0, 0.1])
            else:
                p.resetDebugVisualizerCamera(0.7, 0, -30, [0, 0, 0.1])
            roll += speed
            CC.interaction()
            CC.remember()
            CC.update_states()
    else:
        with writer.as_default():
            for episode in range(Max_episodes):
                CC.reset()
                CC.done = False
                CC.total_reward = 0
                while not CC.done:
                    CC.interaction()
                    CC.remember()
                    CC.learn()
                    CC.fresh_para()
                    CC.update_states()
            total_reward_list.append(CC.total_reward)
            tf.summary.scalar("reward", CC.total_reward, step=episode)
            writer.flush()
            if np.mean(total_reward_list) > best_avg_reward:
                CC.agent1.save_model()
                CC.agent2.save_model()
                CC.agent3.save_model()
                best_avg_reward = np.mean(total_reward_list)
            print("episode:", episode, "score", CC.total_reward)



