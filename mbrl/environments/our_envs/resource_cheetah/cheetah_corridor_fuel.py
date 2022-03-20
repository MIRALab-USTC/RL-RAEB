import numpy as np
import pandas as pd 
import torch

from gym import utils, spaces
import os
import math 
#from ipdb import set_trace
# For test
# import sys
# sys.path.insert(0, '/home/zhwang/research/mbrl_exploration_with_novelty')

from mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor import CheetahCorridor

class CheetahCorridorFuel(CheetahCorridor):
    @classmethod
    def name(cls):
        return "FuelCheetahCorridorEnv"

    def __init__(self, cargo_num, beta, reward_block, reward, alpha):
        self.cargo_num = cargo_num # the number of resources 
        self.cur_cargo = cargo_num
        self.beta = beta
        self.reward = reward
        self.alpha = alpha 
        CheetahCorridor.__init__(self, reward_block)

    def get_state_pos(self, x):
        r_x_low = self.reward_block[0]
        r_x_high = self.reward_block[1]
        if x >= r_x_low:
            return True
        else:
            return False

    def _get_obs(self):
        # add s[0] x position 
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            [self.cur_cargo],
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.cur_cargo = self.cargo_num
        return self._get_obs()

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

    def f_batch(self, states, actions):
        # a\in [0,1] wrapped by env
        # continuous cargo
        actions_cargo = torch.sum(0.1 * torch.square(actions), dim=-1, keepdim=True)
        if not torch.is_tensor(actions_cargo):
            actions_cargo = torch.from_numpy(actions_cargo)
        actions_cargo = actions_cargo.reshape((actions_cargo.shape[0],1)).float()

        states_cargo = states[:,-1]
        if not torch.is_tensor(states_cargo):
            states_cargo = torch.from_numpy(states_cargo)
        states_cargo = states_cargo.reshape((states_cargo.shape[0],1)).float()    
        w = torch.zeros_like(actions_cargo, dtype=actions_cargo.dtype)

        indexes_states = torch.where((states_cargo-actions_cargo).float()<=0)
        w[indexes_states] = states_cargo[indexes_states]

        indexes_actions = torch.where((states_cargo-actions_cargo)>0)
        w[indexes_actions] = actions_cargo[indexes_actions]
        return w    

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # update cargo
        action_cargo = min(0.1 * np.square(action).sum(), self.cur_cargo)
        self.cur_cargo = min(max(self.cur_cargo - action_cargo, 0), self.cargo_num)
        
        obs = self._get_obs()

        done = False
        reward = 0
        cur_x_pos_r = self.get_state_pos(xposafter)
        if cur_x_pos_r:
            done = True
            reward = self.reward + (self.reward * self.cur_cargo / self.cargo_num) 

        return obs, reward, done, dict(action_cargo=action_cargo)

class CheetahCorridorFuelDone(CheetahCorridorFuel):
    @classmethod
    def name(cls):
        return "FuelCheetahCorridorDoneEnv" 

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # update cargo
        action_cargo = min(0.1 * np.square(action).sum(), self.cur_cargo)
        self.cur_cargo = min(max(self.cur_cargo - action_cargo, 0), self.cargo_num)
        
        obs = self._get_obs()

        done = False
        reward = 0
        cur_x_pos_r = self.get_state_pos(xposafter)
        if cur_x_pos_r:
            done = True
            reward = self.reward + (self.reward * self.cur_cargo / self.cargo_num) 

        if self.cur_cargo <= 0:
            done = True

        return obs, reward, done, dict(action_cargo=action_cargo)

if __name__=='__main__':
    # test ant maze env
    #env = ResourceCheetahCorridor(cargo_num=4, beta=5, reward_block=[4,5], reward=100)
    #video_env = VideoEnv(env)
    #state = env.reset()
    pass


