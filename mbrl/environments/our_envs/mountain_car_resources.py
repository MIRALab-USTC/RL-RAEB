# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.spaces import Discrete
from gym.utils import seeding

from mbrl.environments.our_envs.continuous_mountain_car import Continuous_MountainCarEnv


class ResourceMountainCarEnv(Continuous_MountainCarEnv):

  def __init__(self, goal_velocity=0, cargo_num=10.0):
    self.min_action = -1.0
    self.max_action = 1.0
    self.min_position = -1.2
    self.max_position = 0.6
    self.max_speed = 0.07
    self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    self.goal_velocity = goal_velocity
    self.power = 0.0015
    self.viewer = None

    self.cargo_num = cargo_num
    self.cargo = cargo_num

    self.low_state = np.array(
        [self.min_position, -self.max_speed, 0.], dtype=np.float32
    )
    self.high_state = np.array(
        [self.max_position, self.max_speed, self.cargo_num], dtype=np.float32
    )
    self.action_space = spaces.Box(
        low=np.array([self.min_action, 0.]),
        high=np.array([self.max_action, self.cargo_num]),
        dtype=np.float32
    )
    self.observation_space = spaces.Box(
        low=self.low_state,
        high=self.high_state,
        dtype=np.float32
    )
    self.seed()
    self.reset()


  def step(self, action):

    position = self.state[0]
    velocity = self.state[1]
    # the change of the cargo num
    cargo_action = min(max(action[1], 0), self.cargo)
    self.cargo = min(max(self.cargo - cargo_action, 0), self.cargo_num)
    
    force = min(max(action[0], self.min_action), self.max_action)

    velocity += force * self.power - 0.0025 * math.cos(3 * position)
    if (velocity > self.max_speed): velocity = self.max_speed
    if (velocity < -self.max_speed): velocity = -self.max_speed
    position += velocity
    if (position > self.max_position): position = self.max_position
    if (position < self.min_position): position = self.min_position
    if (position == self.min_position and velocity < 0): velocity = 0

    # if and only if the car reach the peak with positive velocity and the cargo is exhausting.
    done = bool(
        position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
    )

    reward = 0
    if done:
        reward = 100 * cargo_action
    # reward -= math.pow(action[0], 2) * 0.1

    self.state = np.array([position, velocity, self.cargo])
    # if self.cargo == 0:
    #   return self.state, 0, done, {}
    return self.state, reward, done, {}

  def reset(self):
    self.cargo = self.cargo_num
    self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0, self.cargo])
    return np.array(self.state)

  def I(self, state):
    return np.array(self.cargo)

  def f(self, state, action):
    cargo_action = min(max(action[1], 0), self.cargo)
    return cargo_action
