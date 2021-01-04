# RL-based-3D-inverted-pendulum-balance-control
ME5406 project2. This project is also a preliminary exploration of applying RL algorithms to a ballbot.

## Introduction
In this project, we explore Reinforcement Learning (RL) based self-balancing control of a 3D inverted pendulum,which is a reaction-wheel-based cube robot. ([The Cubli](https://idsc.ethz.ch/research-dandrea/research-projects/archive/cubli.html))


We apply different algorithms and compare their difference.

First, design agents to control it balance on its edge. Then, on its corner.

## Requirements
gym

pybullet     version == 2.9.3

tensorflow   version == 2.0.0
 
## Results
Traning logs can be found in folders/mylogs

Demo
https://youtu.be/kXEApVPTZNA

https://youtu.be/seaSP5W9LvY
### Balance on edge

#### DQN
(discrete action space)
![image](/gif/DQN.gif)
#### DDQN
(discrete action space)
![image](/gif/DDQN.gif)
#### DDPG
(continuous action space)
![image](/gif/DDPG.gif)

### Balance on corner
#### DDPG
![image](/gif/DDPG_corner.gif)

#### MADDPG
(still under training)
![image](/gif/MADDPG.gif)
