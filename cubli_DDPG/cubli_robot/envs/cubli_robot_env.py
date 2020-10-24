import os
import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

class Cubli_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False):
        self._observation = []
        self.action_space = spaces.Box(low=-10, high=10, shape=(1, ), dtype=np.float32)   # continuous actions
        self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, -200]),
                                            np.array([math.pi, math.pi, 200]), dtype=np.float32)
        # action_space[robot pitch, robot pitch angular velocity, wheel velocity ]

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.wheel_velocity = 0
        self.step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0, 0, 0.1415]        # to set the start position
        cubeStartOrientation = p.getQuaternionFromEuler([math.pi/4, 0, 0])

        load_path = os.path.abspath(os.path.dirname(__file__))
        self.boxId = p.loadURDF(os.path.join(load_path, "cubli_robot.xml"),
                                cubeStartPos, cubeStartOrientation)
        self._observation = self._compute_observation()
        return np.array(self._observation, dtype=np.float32)

    def _step(self, action):
        self._pybullet_control(action)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        self.step_counter += 1
        if done:
            print("steps taken", self.step_counter)
        return np.array(self._observation, dtype=np.float32), reward, done, {}

    def _pybullet_control(self, action):
        self.deltav = action        # problem
        self.wheel_velocity += self.deltav
        p.setJointMotorControl2(bodyUniqueId=self.boxId,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.wheel_velocity)

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.boxId)
        return [cubeEuler[0], angular[0], self.wheel_velocity]       # pitch, pitch angular velocity, wheel velocity

    def _compute_reward(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.boxId)
        return 1-abs(math.pi/4-cubeEuler[0]) - abs(angular[0])*0.1 - abs(cubePos[1])*0.1

    def _compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        return self.step_counter >= 1500 or abs(math.pi/4-cubeEuler[0]) > math.pi/4

    def _render(self, mode='human', close=False):
        pass