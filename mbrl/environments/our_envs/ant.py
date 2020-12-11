'''Adapted from Model-Based Active Exploration (https://github.com/nnaisense/max)'''

import numpy as np
import pandas as pd 

from gym import utils
from gym.envs.mujoco import mujoco_env

import os
import math 

from mujoco_py.generated import const


def get_state_block(state):
    x = state[2].item()
    y = state[3].item()

    if -1 < y < 1:
        y_block = 'low'
    elif 1 < y < 3:
        y_block = 'mid'
    elif 3 < y < 5:
        y_block = 'high'
    else:
        raise Exception

    if -1 < x < 1:
        x_block = 'left'
    elif 1 < x < 3:
        x_block = 'center'
    elif 3 < x < 5:
        x_block = 'right'
    else:
        raise Exception

    if x_block == 'left' and y_block == 'low':
        return 0
    elif x_block == 'center' and y_block == 'low':
        return 1
    elif x_block == 'right' and y_block == 'low':
        return 2
    elif x_block == 'right' and y_block == 'mid':
        return 3
    elif x_block == 'right' and y_block == 'high':
        return 4
    elif x_block == 'center' and y_block == 'high':
        return 5
    elif x_block == 'left' and y_block == 'high':
        return 6

def rate_buffer(buffer):
    visited_blocks = [get_state_block(state) for state in buffer.states]
    n_unique = len(set(visited_blocks))
    return n_unique

class MagellanAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Observation Space: 
        - x torso COM velocity
        - y torso COM velocity
        - 15 joint positions
        - 14 joint velocities
        - (optionally, commented for now) 84 contact forces
    """
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant_maze.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def contact_forces(self):
        return np.clip(self.sim.data.cfrc_ext, -1, 1)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        return obs, 0, False, {}
    

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt

        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        return self._get_obs()

    def viewer_setup(self):
        #self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.type = const.CAMERA_USER
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent
        self.viewer.cam.lookat[0] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -85
        self.viewer.cam.azimuth = 235

class AntMazeEnv(MagellanAntEnv):
    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        done = False
        reward = 0
        if get_state_block(obs) >= 2:
            done = True
            reward = 100
        return obs, reward, done, {}

class AntMazeEnvForwardReward(MagellanAntEnv):
    def __init__(self):
        self.step_id = 0
        self.block = 0
        self.block_array = {"step_id": [0], "block": [0]}
        MagellanAntEnv.__init__(self)
        
    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        #survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost

        obs = self._get_obs()

        #notdone = np.isfinite(obs).all() \
        #    and obs[2] >= 0.2 and obs[2] <= 1.0
        done = False
        if get_state_block(obs)==3:
            done = True
        
        if done or self.step_id >= 500:
            self.step_id = 0

        self.step_id += 1
        self.block = get_state_block(obs)
        self.block_array['step_id'].append(self.step_id)
        self.block_array['block'].append(self.block)
        
        if self.step_id > 400:
            if not os.path.exists("./block.csv"):
                df = pd.DataFrame(self.block_array)
                df.to_csv("/home/rl_shared/zhihaiwang/block.csv")
        
        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost)

class AntMazeEnvGoal(MagellanAntEnv):
    def __init__(self, goal_pos):
        self.step_id = 0
        self.block = 0
        self.block_array = {"step_id": [0], "block": [0]}
        self.goal_pos = goal_pos

        MagellanAntEnv.__init__(self)

    def distance_l2(self, pos_dict):
        d_r = 0
        for key, value in self.goal_pos.items():
            d = (value-pos_dict[key]) ** 2
            d_r += d

        return - math.sqrt(d_r)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        
        # sport reward
        #forward_reward = (xposafter - xposbefore)/self.dt
        #ctrl_cost = .5 * np.square(action).sum()
        #contact_cost = 0.5 * 1e-3 * np.sum(
        #    np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        #survive_reward = 1.0
        #reward = forward_reward - ctrl_cost - contact_cost
        
        # distance reward 

        reward = self.distance_l2(dict(x=xposafter, y=yposafter))

        obs = self._get_obs()

        #notdone = np.isfinite(obs).all() \
        #    and obs[2] >= 0.2 and obs[2] <= 1.0
        done = False
        if get_state_block(obs)==3:
            done = True
        
        if done or self.step_id >= 500:
            self.step_id = 0

        self.step_id += 1
        self.block = get_state_block(obs)
        self.block_array['step_id'].append(self.step_id)
        self.block_array['block'].append(self.block)
        
        if self.step_id > 400:
            if not os.path.exists("./block.csv"):
                df = pd.DataFrame(self.block_array)
                df.to_csv("./block.csv")
        
        return obs, reward, done, {}

class AntMazeEnvGoalForwardReward(AntMazeEnvGoal):

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        
        # sport reward
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        #survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost
        
        # distance reward 

        d_reward = self.distance_l2(dict(x=xposafter, y=yposafter))

        reward += d_reward

        obs = self._get_obs()

        #notdone = np.isfinite(obs).all() \
        #    and obs[2] >= 0.2 and obs[2] <= 1.0
        done = False
        if get_state_block(obs)==3:
            done = True
        
        if done or self.step_id >= 500:
            self.step_id = 0

        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost)


if __name__=='__main__':
    # test ant maze env
    env = AntMazeEnv()

    state = env.reset_model()
    action = env.action_space.sample() # action_space
    _, reward, _, _ = env.step(action)

    print(reward)
