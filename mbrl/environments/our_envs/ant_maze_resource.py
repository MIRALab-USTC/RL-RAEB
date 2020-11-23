'''Adapted from Model-Based Active Exploration (https://github.com/nnaisense/max)'''

import numpy as np

from gym import utils,spaces
from gym.envs.mujoco import mujoco_env

import os

from mujoco_py.generated import const

#from mbrl.environments.our_envs.ant import get_state_block, AntMazeEnv
from ant import get_state_block, AntMazeEnv

class AntMazeResourceEnv(AntMazeEnv):

    def __init__(self, cargo_num):
        self.cargo_num = cargo_num# the number of resources 
        AntMazeEnv.__init__(self)
         
    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action[:-1], self.frame_skip)
        obs = self._get_obs(action[-1])
        reward = self._judge_position(obs, action)
        if reward > 0:
            print(f"reward: {reward}")
        return obs, reward, False, {}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = np.concatenate((low, [0]))
        high = np.concatenate((high,[self.cargo_num]))
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _judge_position(self, state, action):
        # reach the last block and release resources 
        if get_state_block(state) == 6 and min(state[-1], action[-1]) > 0:
            return 100 * min(state[-1], action[-1])
        else:
            return 0

    def _get_obs(self, action_resources):
        # state: raw_state + cargo_num
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt

        cargo_num = max(0, self.cargo_num - action_resources)
        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities, [cargo_num]))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        
        return self._get_obs(0) # action resource = 0 


if __name__=='__main__':
    # test ant maze env
    env = AntMazeResourceEnv(10)

    state = env.reset_model()
    action = env.action_space.sample() # action_space
    print(action.shape)
    
    _, reward, _, _ = env.step(action)

    print(reward)
