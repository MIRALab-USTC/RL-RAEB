# -*- coding: utf-8 -*-

import math
import torch
import copy

import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from gym.utils import seeding
# from gym.envs.classic_control import rendering

from mbrl.environments.our_envs.mountain_car.continuous_mountain_car import ContinuousMountainCarEnv

class FuelCargoMountainCar(ContinuousMountainCarEnv):
    @classmethod
    def name(cls):
        return "FuelCargoMountainCar"

    def __init__(self, seed, beta=1, goal_velocity=0, fuel_num=5.0, cargo_num=4.0, alpha=[1,1]):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.cargo_num = [fuel_num, cargo_num] # 总资源量
        self.cargo = [fuel_num, cargo_num] # 当前资源量

        self.low_state = np.array(
            [self.min_position, -self.max_speed, 0., 0.], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, fuel_num, cargo_num], dtype=np.float32
        )

        # TODO check: action 消耗货物是0-1 吗？ -1,0 代表不消耗资源
        self.action_space = spaces.Box(
            low=np.array([self.min_action, -1.]),
            high=np.array([self.max_action, cargo_num]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.seed(seed)
        self.beta = beta
        self.alpha = alpha
        self.viewer = None 
        self.reset() 

    def reset(self):
        assert self.cargo_num[0] > 0 
        self.cargo = copy.deepcopy(self.cargo_num)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0, self.cargo[0], self.cargo[1]])
        return self.state

    def step(self, action):
        # TODO 根据2个资源的进行修改环境
        # 注意fuel 耗完得done，cargo 卸完还可以不done
        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        # fuel action
        fuel_action = min(0.1 * np.square(action[0]), self.cargo[0])
        # cargo_action
        cargo_action = min(max(action[1], 0), self.cargo[1])

        self.cargo[0] = min(max(self.cargo[0] - fuel_action, 0), self.cargo_num[0])
        self.cargo[1] = min(max(self.cargo[1] - cargo_action, 0), self.cargo_num[1])
        
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        # reach peak
        if reach_peak:
            reward = 100 + (100 * self.cargo[0] / self.cargo_num[0]) + (100 * cargo_action / self.cargo_num[1]) 
            done = True

        # no energy left
        if self.cargo[0] <= 0:
            done = True
        self.state = np.array([position, velocity, self.cargo[0], self.cargo[1]])

        return self.state, reward, done, dict(fuel_action=fuel_action, action_cargo=cargo_action, state_fuel=self.cargo[0], state_cargo=self.cargo[1])

    def I(self, state):
        return state[-2:] # [fuel, cargo]
    
    def I_batch(self, states):
        state_cargo = states[:,-2:]
        state_cargo = state_cargo.reshape((state_cargo.shape[0],2))
        if not torch.is_tensor(state_cargo):
            state_cargo = torch.from_numpy(state_cargo)
        return state_cargo
    
    def get_long_term_weight_batch(self, states, actions):
        I_s = self.I_batch(states)

        # f_s_a = self.f_batch(states, actions)
        w = self.beta * ((self.alpha[0] + I_s[:,:1].float()) / (self.alpha[0] + self.cargo_num[0])) \
            * ((self.alpha[1] + I_s[:,1:2].float()) / (self.alpha[1] + self.cargo_num[1])) 
        return w

    def f_batch(self, states, actions):
        pass