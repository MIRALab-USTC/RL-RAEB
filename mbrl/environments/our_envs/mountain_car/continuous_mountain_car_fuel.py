# -*- coding: utf-8 -*-

import math
import torch

import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from gym.utils import seeding
from gym.envs.classic_control import rendering

from mbrl.environments.our_envs.mountain_car.continuous_mountain_car import ContinuousMountainCarEnv

class FuelMountainCarEnv(ContinuousMountainCarEnv):
    # TODO: 环境有bug，电量耗完之后还能正常执行动作！！！
    @classmethod
    def name(cls):
        return "FuelMountainCarEnv"

    def __init__(self, seed, beta=1, goal_velocity=0, fuel_num=5.0, alpha=1):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.cargo_num = fuel_num
        self.cargo = fuel_num
        self.low_state = np.array(
            [self.min_position, -self.max_speed, 0.], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, self.cargo_num], dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
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
        self.cargo = self.cargo_num
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0, self.cargo])
        return self.state

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        cargo_action = min(0.1 * np.square(action[0]), self.cargo)
        self.cargo = min(max(self.cargo - cargo_action, 0), self.cargo_num)
        assert self.cargo >= 0
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0


        # if and only if the car reach the peak with positive velocity and the cargo is exhausting.
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if reach_peak:
            reward = 100 + 100 * self.cargo
            done = True

        self.state = np.array([position, velocity, self.cargo])

        return self.state, reward, done, dict(action_cargo=cargo_action, state_cargo=self.cargo)

    def I(self, state):
        return state[-1]
    
    def I_batch(self, states):
        state_cargo = states[:,-1]
        state_cargo = state_cargo.reshape((state_cargo.shape[0],1))
        if not torch.is_tensor(state_cargo):
            state_cargo = torch.from_numpy(state_cargo)
        return state_cargo
    
    def get_long_term_weight_batch(self, states, actions):
        I_s = self.I_batch(states)
        # f_s_a = self.f_batch(states, actions)
        w = self.beta * (self.alpha + I_s.float()) / (self.alpha + self.cargo_num)
        return w
class FuelMountainCarR100(FuelMountainCarEnv):
    @classmethod
    def name(cls):
        return "FuelMountainCarR100Env"

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        cargo_action = min(0.1 * np.square(action[0]), self.cargo)
        self.cargo = min(max(self.cargo - cargo_action, 0), self.cargo_num)
        assert self.cargo >= 0
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0


        # if and only if the car reach the peak with positive velocity and the cargo is exhausting.
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if reach_peak:
            reward = 100
            done = True

        self.state = np.array([position, velocity, self.cargo])

        return self.state, reward, done, dict(action_cargo=cargo_action, state_cargo=self.cargo)

class FuelMountainCarR100Done(FuelMountainCarEnv):
    @classmethod
    def name(cls):
        return "FuelMountainCarR100Env"

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        cargo_action = min(0.1 * np.square(action[0]), self.cargo)
        self.cargo = min(max(self.cargo - cargo_action, 0), self.cargo_num)
        assert self.cargo >= 0
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0


        # if and only if the car reach the peak with positive velocity and the cargo is exhausting.
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reward = 0
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        if reach_peak:
            reward = 100
            done = True

        # no energy left
        if self.cargo <= 0:
            done = True
        self.state = np.array([position, velocity, self.cargo])

        return self.state, reward, done, dict(action_cargo=cargo_action, state_cargo=self.cargo)

class FuelMountainCarDone(FuelMountainCarEnv):
    @classmethod
    def name(cls):
        return "FuelMountainCarDoneEnv"

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        cargo_action = min(0.1 * np.square(action[0]), self.cargo)
        self.cargo = min(max(self.cargo - cargo_action, 0), self.cargo_num)
        assert self.cargo >= 0
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0


        # if and only if the car reach the peak with positive velocity and the cargo is exhausting.
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reward = 0
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        if reach_peak:
            reward = 100 + (100 * self.cargo / self.cargo_num)
            done = True

        # no energy left
        if self.cargo <= 0:
            done = True
        self.state = np.array([position, velocity, self.cargo])

        return self.state, reward, done, dict(action_cargo=cargo_action, state_cargo=self.cargo)

    