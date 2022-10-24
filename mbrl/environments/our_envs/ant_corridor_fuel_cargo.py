'''Adapted from Model-Based Active Exploration (https://github.com/nnaisense/max)'''

import numpy as np
import pandas as pd 
import torch

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

import os
import math 
import copy

from mujoco_py.generated import const
from mbrl.environments.our_envs.ant_corridor import AntCorridorEnv

class AntCorridorFuelCargo(AntCorridorEnv):
    def __init__(
        self,
        fuel_num,
        cargo_num,
        beta,
        reward_block,
        reward,
        alpha
    ):
        self.cargo_num = [fuel_num, cargo_num]
        self.cur_cargo = [fuel_num, cargo_num]
        self.beta = beta
        self.reward = reward
        self.alpha = alpha
        AntCorridorEnv.__init__(self, reward_block)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = np.concatenate((low, [0]))
        high = np.concatenate((high,[self.cargo_num[1]]))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def get_state_pos(self, state):
        x = state[2].item()
        r_x_low = self.reward_block[0]
        r_x_high = self.reward_block[1]
        if x >= r_x_low and x <= r_x_high:
            return True
        else:
            return False

    def _get_obs(self):
        # state: raw_state + fuel_num + cargo_num
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt

        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities, [self.cur_cargo[0]], [self.cur_cargo[1]]))

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action[:-1], self.frame_skip)
        # update fuel 
        fuel_action = min(0.1 * np.square(action[:-1]).sum(), self.cur_cargo[0])
        self.cur_cargo[0] = min(max(self.cur_cargo[0] - fuel_action, 0), self.cargo_num[0])
        
        # update cargo
        # the cargo of last state
        cargo_last = self.cur_cargo[1]
        action_cargo = action[-1]
        if action_cargo > 0.5:
            self.cur_cargo[1] = max(0, self.cur_cargo[1] - 1)
        
        # update obs 
        obs = self._get_obs()

        cargo_now = self.cur_cargo[1]
        done = False
        reward = 0
        cur_x_pos_r = self.get_state_pos(obs)
        if cur_x_pos_r and cargo_now < cargo_last:
            done = True
            reward = self.reward + (self.reward * self.cur_cargo[0] / self.cargo_num[0]) 

        if self.cur_cargo[0] <= 0: # no fuel
            done = True
            
        return obs, reward, done, dict(state_fuel=self.cur_cargo[0], state_cargo=self.cur_cargo[1], action_cargo=action_cargo, fuel_action=fuel_action)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.cur_cargo = copy.deepcopy(self.cargo_num)
        obs = self._get_obs() # action resource = 0 
        obs[-2] = self.cargo_num[0]
        obs[-1] = self.cargo_num[1]
        return obs

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
        # a\in [0,1] wrapped by env
        # continuous cargo
        fuel_action = 0.1 * torch.sum(torch.square(actions[:,:-1]), axis=1)
        fuel_action = fuel_action.reshape(fuel_action.shape[0], 1)
        cargo_action = actions[:,-1:]

        return fuel_action + cargo_action
    


