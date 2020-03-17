
import numpy as np
import itertools
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

from collections import deque
from mbrl.environments.utils import make_gym_env, set_env_seed

class ProxyEnv(Env):
    def __init__(self, env_name):
        self.env_name = env_name
        self._build_wrapped_envs() 
        self.observation_shape = self.observation_space.shape
        self.action_shape = self.action_space.shape
    
    def _build_wrapped_envs(self, seed=None):
        self._wrapped_env = make_gym_env(self.env_name)
        set_env_seed(self._wrapped_env, seed)
        self.observation_space = self._wrapped_env.observation_space
        self.action_space = self._wrapped_env.action_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)
