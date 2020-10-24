from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, length):
        self.Memory = deque(maxlen=length)

    def store(self, state, action1, action2, action3, reward, next_state, done):
        self.Memory.append((state, action1, action2, action3, reward, next_state, done))

    def sample(self, batch_size):
        batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done = zip(
            *random.sample(self.Memory, batch_size))

        batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done = \
            [np.array(a, dtype=np.float32) for a in [batch_state, batch_action1, batch_action2, batch_action3,
                                                     batch_reward, batch_next_state, batch_done]]
        return batch_state, batch_action1, batch_action2, batch_action3, batch_reward, batch_next_state, batch_done
