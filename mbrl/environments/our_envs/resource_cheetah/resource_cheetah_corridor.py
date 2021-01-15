import numpy as np
import pandas as pd 
import torch

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

import os
import math 

from mujoco_py.generated import const
#from ipdb import set_trace
# For test
import sys
sys.path.insert(0, '/home/zhwang/research/mbrl_exploration_with_novelty')


from mbrl.environments.video_env import VideoEnv

class CheetahCorridor(HalfCheetahEnv):
    def __init__(self, reward_block):
        self.reward_block = reward_block
        #HalfCheetahEnv.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/half_cheetah_corridor.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        #reward_ctrl = - 0.1 * np.square(action).sum()
        #reward_run = (xposafter - xposbefore)/self.dt
        #reward = reward_ctrl + reward_run
        reward = 0
        done = False
        x = xposafter
        if x > self.reward_block[0]:
            reward = 100
            done = True
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer.cam.lookat[0] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 270

class ResourceCheetahCorridor(CheetahCorridor):
    def __init__(self, cargo_num, beta, reward_block, reward):
        self.cargo_num = cargo_num # the number of resources 
        self.cur_cargo = cargo_num
        self.beta = beta
        self.reward = reward
        CheetahCorridor.__init__(self, reward_block)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = np.concatenate((low, [0]))
        high = np.concatenate((high,[self.cargo_num]))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def get_state_pos(self, x):
        r_x_low = self.reward_block[0]
        r_x_high = self.reward_block[1]
        if x >= r_x_low and x <= r_x_high:
            return True
        else:
            return False

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action[:-1], self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        # update cargp
        action_cargo = action[-1]
        # the cargo of last state
        cargo_last = self.cur_cargo

        if action_cargo > 0.5:
            # consume resource
            self.cur_cargo = max(0, self.cur_cargo - 1)
            # cargo_num = max(0, self.cargo_num - 1)

        obs = self._get_obs()

        cargo_now = self.cur_cargo
        done = False
        reward = 0
        cur_x_pos_r = self.get_state_pos(xposafter)
        if cur_x_pos_r and cargo_now < cargo_last:
            done = True
            reward = self.reward
        #reward_ctrl = - 0.1 * np.square(action).sum()
        #reward_run = (xposafter - xposbefore)/self.dt
        #reward = reward_ctrl + reward_run

        return obs, reward, done, dict(action_cargo=action_cargo)

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

    def get_long_term_weight_batch(self, states, actions):
        I_s = self.I_batch(states)
        f_s_a = self.f_batch(states, actions)
        w = self.beta * (1 + I_s.float() - f_s_a.float()) / (1 + self.cargo_num)
        return w


    def I_batch(self, states):
        state_cargo = states[:,-1]
        state_cargo = state_cargo.reshape((state_cargo.shape[0],1))
        if not torch.is_tensor(state_cargo):
            state_cargo = torch.from_numpy(state_cargo)
        return state_cargo
    
    def f_batch(self, states, actions):
        # a\in [0,1] wrapped by env
        actions_cargo = actions[:,-1]
        if not torch.is_tensor(actions_cargo):
            actions_cargo = torch.from_numpy(actions_cargo)
        actions_cargo = actions_cargo.reshape((actions_cargo.shape[0],1))
        actions_cargo = actions_cargo * 2 - 1
        
        w = torch.sign(actions_cargo) 
        zero = torch.zeros_like(w)
        w = torch.where(w <= 0, zero, w)

        # state cargo is 0 
        states_cargo = states[:,-1]
        if not torch.is_tensor(states_cargo):
            states_cargo = torch.from_numpy(states_cargo)
        indexes_invalid = torch.where(states_cargo==0)
        w[indexes_invalid] = 0
        return w




if __name__=='__main__':
    # test ant maze env
    #env = ResourceCheetahCorridor(cargo_num=4, beta=5, reward_block=[4,5], reward=100)
    #video_env = VideoEnv(env)
    #state = env.reset()
    #q_pos = np.zeros(9)
    #q_pos[0] = 4.5
    #env.set_state(q_pos, q_vel)
    #action = env.action_space.sample()

    #o, r, done ,_ = env.step(action)
    #print(o[0])
    #print(done)
    # env_name = "ant_corridor_resource_env_goal_7_v0"
    # video_env = AntCorridorEnv([7,8])

    LEN = 1000
    dire = os.path.join(os.getcwd(),"videos")
    print(dire)
    video_env = VideoEnv("resource_cheetah_corridor_v0", directory=dire)

    state = video_env.reset()
    action = video_env.action_space.sample()
    print(state.shape)
    print(action.shape)
    #set_trace()
    states = np.repeat(state, 3, axis=0)
    actions = np.repeat(np.expand_dims(action, axis=0), 3, axis=0)
    print(states.shape)
    print(actions.shape)
    w = video_env.get_long_term_weight_batch(states, actions)

    for i in range(LEN):
         action = video_env.action_space.sample() # action_space
         next_o, _, _, _ = video_env.step(action)


