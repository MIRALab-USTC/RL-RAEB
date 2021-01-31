# -*- coding: utf-8 -*-

import math
import torch

import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from gym.utils import seeding
from gym.envs.classic_control import rendering

# For test
import sys
sys.path.insert(0, '/home/zhwang/research/mbrl_exploration_with_novelty')

from mbrl.environments.our_envs.mountain_car.continuous_mountain_car import ContinuousMountainCarEnv

from ipdb import set_trace

class ResourceMountainCarEnv(ContinuousMountainCarEnv):

    @classmethod
    def name(cls):
        return "ResourceMountainCarEnv"

    def __init__(self, seed, beta=5, goal_velocity=0, cargo_num=2.0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
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
            low=np.array([self.min_action, -1.]),
            high=np.array([self.max_action, self.cargo_num]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self._spec = None
        self.reward_range = 200

        self.beta = beta 

        self.seed(seed)
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
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if reach_peak:
            reward = 100 * cargo_action
        #reward -= math.pow(action[0], 2) * 0.1
        if reach_peak and self.cargo == 0:
            done = True

        self.state = np.array([position, velocity, self.cargo])
        if cargo_action > 0:
            self.cargo_recorder.append({'cargo': cargo_action, 'position': self.state[0]})
        self.cargo_recorder[0] = {'cargo': self.cargo, 'position': self.state[0]}

        return self.state, reward, done, dict(action_cargo=cargo_action)

    def reset(self):
        self.cargo = self.cargo_num
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0, self.cargo])
        self.cargo_recorder = [{'cargo': self.cargo, 'position': self.state[0]}]  # the first record must be the remaining cargo
        return np.array(self.state)

    def I(self, state):
        return state[-1]

    def f(self, state, action):
        cargo = state[-1]
        cargo_action = min(max(action[1], 0), cargo)
        return cargo_action

    def render(self, mode='human'):
        screen_width = 6000
        screen_height = 4000

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 400
        carheight = 200
        cargo_width = 300
        cargo_height = 150
        clearance = 100

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)
            
            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))
        self.cargo_viewer(carheight, cargo_height, cargo_width, clearance, scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def cargo_viewer(self, carheight, cargo_height, cargo_width, clearance, scale):
        for i, cargo_item in enumerate(self.cargo_recorder):
            cargo_num, pos = cargo_item['cargo']/self.cargo_num, cargo_item['position']
            l, r, t, b = -cargo_width / 2, cargo_width / 2, cargo_height * cargo_num, 0 
            cargo = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cargo.set_color(0.1171875, 0.5625, 1.0)
            on_car_flag = i == 0 and pos == self.state[0]
            cargo.add_attr(rendering.Transform(translation=(0, (clearance + carheight) * on_car_flag)))
            cargotrans = rendering.Transform()
            cargotrans.set_translation(
                (pos-self.min_position) * scale, self._height(pos) * scale
            )
            cargotrans.set_rotation(math.cos(3 * pos))
            cargo.add_attr(cargotrans)
            self.viewer.add_onetime(cargo)

    def env_info(self):
        info = {
            "goals": self.goals,
            "map": np.array((self.min_position, self.max_position)),
            "type": "line",
            "map_image": np.zeros(50)
        }
        return info


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
        # continuous cargo
        actions_cargo = actions[:,-1]
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

class DiscreteResourceMountainCarEnv(ResourceMountainCarEnv):
    @classmethod
    def name(cls):
        return "DiscreteResourceMountainCarEnv"

    def step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        # the change of the cargo num
        cargo_action = 1 if action[1] > 0 else 0
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
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0

        if reach_peak:
            reward = 100 * cargo_action
        #reward -= math.pow(action[0], 2) * 0.1
        if reach_peak and self.cargo == 0:
            done = True

        self.state = np.array([position, velocity, self.cargo])
        if cargo_action > 0:
            self.cargo_recorder.append({'cargo': cargo_action, 'position': self.state[0]})
        self.cargo_recorder[0] = {'cargo': self.cargo, 'position': self.state[0]}

        return self.state, reward, done, dict(action_cargo=action[1], action_cargo_real=cargo_action)

class DoneResourceMountainCarEnv(ResourceMountainCarEnv):
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
        # done = bool(
        #    position >= self.goal_position and velocity >= self.goal_velocity and self.cargo == 0
        # )
        done = False
        reach_peak = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if reach_peak:
            reward = 100 * cargo_action
        #reward -= math.pow(action[0], 2) * 0.1
        if reach_peak and self.cargo == 0:
            done = True
        
        if self.cargo <= 0:
            done = True

        self.state = np.array([position, velocity, self.cargo])
        if cargo_action > 0:
            self.cargo_recorder.append({'cargo': cargo_action, 'position': self.state[0]})
        self.cargo_recorder[0] = {'cargo': self.cargo, 'position': self.state[0]}

        return self.state, reward, done, dict(action_cargo=cargo_action)


if __name__=='__main__':
    # test ant maze env
    
    import cv2
    #set_trace()
    #env = ResourceMountainCarEnv(seed=None)
    env = gym.make("HalfCheetah-v2")
    
    #set_trace()
    #state = env.reset()
    #action = env.action_space.sample()

    

    env_image = env.render(mode="rgb_array")
    
    print(env_image.shape)
    #cv2.imwrite("ResourceMountainCar3.png", env_image)

    #b = np.ones((10,10), dtype=np.uint8)

    #scale_image = np.kron(env_image, b)
    #print(scale_image.shape)
    #cv2.imwrite("ResourceMountainCar7.jpeg", env_image)

    from PIL import Image
    #RGBimage = cv2.cvtColor(env_image, cv2.COLOR_BGR2RGB)
    PILimage = Image.fromarray(env_image)
    PILimage.save('ant.jpeg', quality=100)