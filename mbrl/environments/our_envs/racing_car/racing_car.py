#!/usr/bin/env python
# -*- coding: utf-8 -*-


from abc import ABC
#For test
#import sys
#sys.path.insert(0, "/home/zhwang/research/mbrl_exploration_with_novelty")

from mbrl.environments.our_envs.mountain_car.base_env import BaseEnv

import numpy as np
from gym import spaces

import torch

import math



class Racing(BaseEnv, ABC):

    @classmethod
    def name(cls):
        return "racing"

    def __init__(self, seed, beta=10, oil=19, total_distance=500, f=0.1, a=0.1, F=0.2):
        """
        简化版的赛车任务。action空间为[0,1]。0表示匀速，1表示加速，加速时消耗的油量更大。
        任务的目标是在油量允许的情况下使用尽量少的step完成任务达到终点。容易证明：
            1. 油量非常充足时，最优策略为一直加速
            2. 油量不够充足时，最优策略为先加速后匀速，且到达终点时油量耗尽。
        Args:
            seed: 随机种子，详见父类定义
            oil: 初始油量
            total_distance: 路面距离
            f: 匀速时每个step消耗的油量，正比于摩擦力
            a: 加速时的加速度
            F: 加速时每个step消耗的油量，正比于牵引力
        TODO：
            1. 定义 action_space, observation_space 等
            2. 实现 render 函数
            3. 根据数学计算得出理论上的最优解
        """
        super().__init__(seed)
        self._oil = oil
        self.beta = oil
        self._init_oil = oil
        self._total_distance = total_distance
        assert 0 < f < F and 0 < a, "Wrong init parameter"
        self._f = f
        self._a = a
        self._F = F
        self._oil = None
        self._finished_distance = None
        self._steps = None
        self._speed = None
        # compute by always accelerating
        self.max_speed = math.sqrt(2 * total_distance * a)
        self.reset()

        self.low_state = np.array(
            [0., 0. , 0.], dtype=np.float32
        )
        self.high_state = np.array(
            [oil, self.max_speed, total_distance], dtype=np.float32
        )
        # action_space: [-1,1] for continuous alg
        self.action_space = spaces.Box(
            low=np.array([-1.]),
            high=np.array([1.]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )


    def step(self, action):
        self._steps += 1
        if action >= 0:
            self._finished_distance += (self._speed + 0.5 * self._a)
            self._speed += self._a
            self._oil -= self._F
        elif action <= 0:
            self._oil -= self._f
            self._finished_distance += self._speed
        else:
            raise Exception("Wrong Action: {}".format(action))
        
        if self._oil <= 0:   
            reward = -1
            done = True
            msg = "Oil exhausted before destination"
        elif self._finished_distance >= self._total_distance:
            reward = 100
            done = True
            msg = "Successful finished the task in {} steps".format(self._steps)    
        else:
            reward = -1
            done = False
            msg = "Still racing"
        next_state = self.state()
        return next_state, reward, done, dict(action_oil=action)

    def state(self):
        return np.array([self._oil, self._speed, self._finished_distance])
        # return np.array([self._oil, self._speed, self._total_distance - self._finished_distance])

    def reset(self):
        self._speed = 0
        self._oil = self._init_oil
        self._finished_distance = 0
        self._steps = 0
        return self.state()

    def I(self, states):
        shape = states.shape
        if len(shape) == 0 or shape[-1] != 3:
            raise Exception("Wrong shape for states: {}".format(shape))
        idx = list(range(len(shape)))
        idx[0], idx[-1] = idx[-1], idx[0]
        # idx: [n-1, 1, 2, 3,,,,n-2,0]
        states.transpose(idx)
        states = states[1:2]
        states.transpose(idx)
        return states

    def f(self, states: np, actions: np):
        states_shape = states.shape
        action_shape = actions.shape
        if len(states_shape) == 0 or states_shape[-1] != 3 or states_shape[:-1] != action_shape:
            raise Exception("Wrong shape for states and actions {}, {}", states_shape, action_shape)
        return action_shape * (self._F - self._f) + self._f

    def render(self, mode='human'):
        pass

    def get_long_term_weight_batch(self, states, actions):
        I_s = self.I_batch(states)
        f_s_a = self.f_batch(states, actions)
        w = self.beta * (1 + I_s.float() - f_s_a.float()) / (1 + self._oil)
        return w


    def I_batch(self, states):
        state_cargo = states[:,0]
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
        
        w = torch.sign(actions_cargo) 
        zero = torch.zeros_like(w)
        w = torch.where(w <= 0, zero, w)
        # 0,1 \to 1,2
        w += 1
        w = 0.1 * w
            
        # state cargo is 0 
        states_cargo = states[:,0]

        if not torch.is_tensor(states_cargo):
            states_cargo = torch.from_numpy(states_cargo)
        indexes_invalid = torch.where(states_cargo==0)
        w[indexes_invalid] = 0
        return w


class RacingSparseReward(Racing):
    
    def step(self, action):
        self._steps += 1
        if action >= 0:
            self._finished_distance += (self._speed + 0.5 * self._a)
            self._speed += self._a
            self._oil -= self._F
        elif action <= 0:
            self._oil -= self._f
            self._finished_distance += self._speed
        else:
            raise Exception("Wrong Action: {}".format(action))
        
        if self._oil <= 0:   
            reward = 0
            done = True
            msg = "Oil exhausted before destination"
        elif self._finished_distance >= self._total_distance:
            reward = 200 - self._steps
            done = True
            msg = "Successful finished the task in {} steps".format(self._steps)    
        else:
            reward = 0
            done = False
            msg = "Still racing"
        next_state = self.state()
        return next_state, reward, done, dict(action_oil=action)


if __name__ == '__main__':
    import sympy
    def get_args(total_distance, f, F, a):
        time_stop_speed_min_oil = math.sqrt(2*f*total_distance/(a * (2*F - f)))
        time_stop_speed_min_oil_square = time_stop_speed_min_oil**2 
        min_oil = F * time_stop_speed_min_oil + f * (2*total_distance - a*time_stop_speed_min_oil_square) / (2 * a * time_stop_speed_min_oil) 
        time_total = (total_distance + 0.5 * a * time_stop_speed_min_oil_square) / (a * time_stop_speed_min_oil)

        print(f"time_stop_speed_min_oil: {time_stop_speed_min_oil}")
        print(f"min_oil: {min_oil}")
        print(f"time_total: {time_total}")
    
    def solve_eq(oil, total_distance, f, F, a):
        x = sympy.symbols('x')
        t1, t2 = sympy.solve(sympy.Eq((2*a*F-a*f) * x**2 - 2*a *oil *x + 2*total_distance*f), x)
        print(f"t1, t2: {t1}, {t2}")

    get_args(500, 0.1, 0.2, 0.1)
    solve_eq(17.5, 500, 0.1, 0.2, 0.1)