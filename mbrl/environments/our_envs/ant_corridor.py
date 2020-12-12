'''Adapted from Model-Based Active Exploration (https://github.com/nnaisense/max)'''

import numpy as np
import pandas as pd 
import torch

from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

import os
import math 

from mujoco_py.generated import const
# For test
# import sys
# sys.path.insert(0, '/home/rl_shared/zhihaiwang/research/mbrl_sparse_reward')

from mbrl.environments.our_envs.ant import MagellanAntEnv
# For test
#from ant import MagellanAntEnv
#from mbrl.environments.video_env import VideoEnv


class AntCorridorEnv(MagellanAntEnv):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant_corridor.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

class AntCorridorResourceEnv(AntCorridorEnv):
    def __init__(self, cargo_num, beta, reward_block):
        self.cargo_num = cargo_num # the number of resources 
        self.cur_cargo = cargo_num
        self.beta = beta
        self.reward_block = reward_block
        AntCorridorEnv.__init__(self)

    def get_state_pos(self, state):
        x = state[2].item()
        r_x_low = self.reward_block[0]
        r_x_high = self.reward_block[1]
        if x >= r_x_low and x <= r_x_high:
            return True
        else:
            return False

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action[:-1], self.frame_skip)
        action_cargo = action[-1]

        # the resource of last state
        cargo_last = self.cur_cargo
        
        # update resource 
        obs = self._get_obs(action_cargo)

        cargo_now = self.cur_cargo
        done = False
        reward = 0
        cur_x_pos_r = self.get_state_pos(obs)
        if cur_x_pos_r and cargo_now < cargo_last:
            done = True
            reward = 100
        
        return obs, reward, done, dict(action_cargo=action_cargo)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = np.concatenate((low, [0]))
        high = np.concatenate((high,[self.cargo_num]))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _get_obs(self, action_resources):
        # state: raw_state + cargo_num
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt
        
        # action > 0 release resource
        # else keep resource
        # discrete resorces
        # assert action_resources <= 1, "action resources out bound"
        if action_resources >= 0.5:
            self.cur_cargo = max(0, self.cur_cargo - 1)
            # cargo_num = max(0, self.cargo_num - 1)
            cargo_num = self.cur_cargo
        else:
            cargo_num = self.cur_cargo

        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities, [cargo_num]))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        obs = self._get_obs(0) # action resource = 0 
        obs[-1] = self.cargo_num
        self.cur_cargo = self.cargo_num
        return obs

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        #self.viewer.cam.type = const.CAMERA_USER
        #self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[0] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 270

        #self.viewer.cam.distance = self.model.stat.extent * 0.5
        #self.viewer.cam.trackbodyid = 1
        #self.viewer.cam.distance = self.model.stat.extent * 1.0
        #self.viewer.cam.lookat[2] = 0.8925
        #self.viewer.cam.elevation = -20

        # self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        # self.viewer.cam.distance = self.model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
        # self.viewer.cam.lookat[0] += 0.5         # x,y,z offset from the object (works if trackbodyid=-1)
        # self.viewer.cam.lookat[1] += 0.5
        # self.viewer.cam.lookat[2] += 0.5
        # self.viewer.cam.elevation = 0           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        # self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis


    def f_batch(self, states, actions):
        actions_cargo = actions[:,-1]
        states_cargo = states[:,-1]
        if not torch.is_tensor(actions_cargo):
            actions_cargo = torch.from_numpy(actions_cargo)
        if not torch.is_tensor(states_cargo):
            states_cargo = torch.from_numpy(states_cargo)
        actions_cargo = actions_cargo.reshape((actions_cargo.shape[0],1))
        w = torch.sign(actions_cargo)
        zero = torch.zeros_like(w)
        w = torch.where(w <= 0, zero, w)
        w = self.beta * w
        one = torch.ones_like(w)
        w = torch.where(w <= 0, one, w)

        # state cargo is 0 
        indexes_invalid = torch.where(states_cargo==0)
        w[indexes_invalid] = 1
        return w

if __name__=='__main__':
    # test ant maze env
    env_name = "ant_corridor_resource_env_goal_7_v0"
    video_env = VideoEnv(env_name, "./videos")

    LEN = 200
    video_env.set_video_name("test")
    o = video_env.reset()
    for i in range(LEN):
        action = video_env.action_space.sample() # action_space
        next_o, _, _, _ = video_env.step(action)


