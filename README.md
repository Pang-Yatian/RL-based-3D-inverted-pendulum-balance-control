# RL-based-3D-inverted-pendulum-balance-control
This project is a preliminary exploration of applying RL algorithms to a ballbot.


Related project: [A Ball Balancing Bobot](https://github.com/Pang-Yatian/A-Ball-Balacicng-Robot) 


Not completely finished yet.

## Introduction
In this project, we explore Reinforcement Learning (RL) based self-balancing control of a 3D inverted pendulum,which is a reaction-wheel-based cube robot. ([The Cubli](https://idsc.ethz.ch/research-dandrea/research-projects/archive/cubli.html))


We apply different algorithms and compare their difference.

First, design agents to control it balance on its edge. Then, on its corner.

## Requirements
gym

pybullet     version == 2.9.3

tensorflow   version == 2.0.0
 
## Results

Demo
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
