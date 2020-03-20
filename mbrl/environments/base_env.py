import random
import abc
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from mbrl.environments.utils import make_gym_env

class MbrlEnv(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env_name):
        pass
    
    @abc.abstractproperty
    def horizon(self):
        pass


class SimpleEnv(MbrlEnv):
    def __init__(self, 
                 env_name,
                 reward_scale=1.0,
                 max_length=np.inf):
        self.env_name = env_name
        self.cur_seed = random.randint(0,65535)
        inner_env = make_gym_env(env_name, self.cur_seed)
        Wrapper.__init__(self, inner_env)
        self.reward_scale = reward_scale
        self.max_length = max_length
        self.low = np.maximum(self.env.action_space.low, -10)
        self.high = np.minimum(self.env.action_space.high, 10)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
    
    @property
    def horizon(self):
        return self.max_length
    
    def reset(self):
        self.cur_step_id = 0
        return np.array([self.env.reset()])

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        action = action[0]
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        if len(action.shape) == len(self.action_space.shape):
            action = np.array([action])
        o, r, d, info = self.env.step(action)
        if self.cur_step_id >= self.max_length:
            done = 1.0
        o, r, d = np.array([o]), np.array([[r]]), np.array([[d]])
        for k in info:
            info[k] = np.array([[info[k]]])
        return o, r, d, info

