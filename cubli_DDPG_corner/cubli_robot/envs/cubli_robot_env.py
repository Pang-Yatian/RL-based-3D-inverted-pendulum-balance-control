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
        self.action_space = spaces.Box(np.array([-10, -10, -10]),
                                       np.array([10, 10, 10]), dtype=np.float32)   # continuous actions

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
        self.wheel_velocity1 = 0
        self.wheel_velocity2 = 0
        self.wheel_velocity3 = 0
        self.step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0, 0, 0.17321]        # to set the start position
        cubeStartOrientation = p.getQuaternionFromEuler([math.pi/4, -math.pi*35.3/180, 0])

        load_path = os.path.abspath(os.path.dirname(__file__))
        self.boxId = p.loadURDF(os.path.join(load_path, "cubli_robot.xml"),
                                cubeStartPos, cubeStartOrientation)
        self._observation = self._compute_observation()
        return np.array(self._observation, dtype=np.float32)


    def _step(self, action):
        action1, action2, action3 = action[0], action[1], action[2]
        self._pybullet_control(action1, action2, action3)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()
        self.step_counter += 1
        if done:
            print("done, steps taken", self.step_counter)
        return np.array(self._observation, dtype=np.float32), reward, done, {}


    def _pybullet_control(self, action1, action2, action3):
        self.deltav1 = action1        # problem
        self.wheel_velocity1 += self.deltav1
        self.deltav2 = action2        # problem
        self.wheel_velocity2 += self.deltav2
        self.deltav3 = action3        # problem
        self.wheel_velocity3 += self.deltav3
        p.setJointMotorControl2(bodyUniqueId=self.boxId,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.wheel_velocity1)
        p.setJointMotorControl2(bodyUniqueId=self.boxId,
                                jointIndex=1,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.wheel_velocity2)
        p.setJointMotorControl2(bodyUniqueId=self.boxId,
                                jointIndex=2,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.wheel_velocity3)

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.boxId)
        return [cubeEuler[0], cubeEuler[1], cubeEuler[2], angular[0], angular[1], angular[2],
                self.wheel_velocity1, self.wheel_velocity2, self.wheel_velocity3]
                                                             # pitch, pitch angular velocity, wheel velocity
                                                               # roll & yaw

    def _compute_reward(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.boxId)
        return 2 - abs(math.pi / 4 - cubeEuler[0]) - abs(angular[0]) * 0.1 \
                - abs(math.pi*35.3/180 + cubeEuler[1]) - abs(angular[1]) * 0.1 - abs(angular[2]) * 0.1

    def _compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        return self.step_counter >= 1500 or abs(math.pi/4-cubeEuler[0]) > math.pi/4 or \
               cubeEuler[1] < -math.pi/2 or cubeEuler[1] > 0

    def _render(self, mode='human', close=False):
        pass