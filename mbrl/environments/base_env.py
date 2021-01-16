import random
import abc
import numpy as np
from gym.spaces import Discrete
from gym import Wrapper
from gym.spaces import Box
from mbrl.environments.utils import make_gym_env
from mbrl.utils.mean_std import RunningMeanStd

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
                 should_normalize_obs=False,
                 reward_scale=1.0,
                 max_length=np.inf):
        self.env_name = env_name
        self.cur_seed = random.randint(0,65535)
        inner_env = make_gym_env(env_name, self.cur_seed)
        Wrapper.__init__(self, inner_env)
        self.reward_scale = reward_scale
        self.max_length = max_length
        if isinstance(self.env.action_space, Box):
            self.low = np.maximum(self.env.action_space.low, -1)
            self.high = np.minimum(self.env.action_space.high, 1)
            ub = np.ones(self.env.action_space.shape)
            self.action_space = Box(-1 * ub, ub)
        elif isinstance(self.env.action_space, Discrete):
            self.low = 0
            self.high = self.env.action_space.n - 1
            self.action_space = Discrete(self.env.action_space.n)

        self.should_normalize_obs = should_normalize_obs
        if should_normalize_obs:
            self.obs_mean_std = RunningMeanStd(self.observation_space.shape)


    def _normalize_observation(self, obs):
        return (obs - self.obs_mean_std.mean) / np.sqrt(self.obs_mean_std.var + 1e-12)

    @property
    def horizon(self):
        return self.max_length
    
    def reset(self):
        self.cur_step_id = 0
        obs = self.env.reset()
        if self.should_normalize_obs:
            self.obs_mean_std.update(obs)
            obs = self._normalize_observation(obs)
        return np.array([obs])

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        action = action[0]
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        o, r, d, info = self.env.step(action)
        if self.should_normalize_obs:
            self.obs_mean_std.update(o)
            o = self._normalize_observation(o)

        if self.cur_step_id >= self.max_length:
            done = 1.0
        o, r, d = np.array([o]), np.array([[r]]), np.array([[d]])
        for k in info:
            info[k] = np.array([[info[k]]])
        return o, r, d, info


class DelayRewardEnv(SimpleEnv):
    def __init__(self, 
                 env_name,
                 steps_delay=20,
                 reward_scale=1.0,
                 max_length=np.inf):
        SimpleEnv.__init__(
            self, 
            env_name,
            reward_scale=1.0,
            max_length=np.inf
        )
        self.step_delay_count = 0
        self.steps_delay = steps_delay
        self.delayed_reward = 0

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        action = action[0]
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        o, r, d, info = self.env.step(action)

        if self.step_delay_count > self.steps_delay:
            r += self.delayed_reward 
            self.step_delay_count = 0
            self.delayed_reward = 0
        else:
            self.step_delay_count += 1
            self.delayed_reward += r
            r = 0
        
        if self.cur_step_id >= self.max_length:
            done = 1.0
        o, r, d = np.array([o]), np.array([[r]]), np.array([[d]])
        for k in info:
            info[k] = np.array([[info[k]]])
        return o, r, d, info