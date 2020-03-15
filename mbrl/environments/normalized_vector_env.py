import numpy as np
import gym
from gym.spaces import Box
import random
from mbrl.environments.utils import make_gym_env, set_env_seed
from mbrl.environments.base_env import ProxyEnv
from mbrl.environments.reward_done_functions import get_reward_done_function
from mbrl.utils.mean_std import RunningMeanStd

class NormalizedVectorEnv(ProxyEnv):
    def __init__(self, 
                 env_name,
                 n_env=1, 
                 reward_scale=1.0,
                 normalize=False,
                 max_length=1000, 
                 must_provide=None):
        self.n_env = n_env
        super(NormalizedVectorEnv, self).__init__(env_name)
        self.reward_scale = reward_scale
        self.max_length = max_length
        self.low = np.maximum(self._wrapped_envs[0].action_space.low, -10)
        self.high = np.minimum(self._wrapped_envs[0].action_space.high, 10)
        self.reward_f, self.done_f = get_reward_done_function(env_name, must_provide)
        self.normalize = normalize
        if normalize:
            self.obs_mean_std = RunningMeanStd(self.observation_shape)
        self.reset()
    
    def _normalize_obs(self, obs):
        return (obs - self.obs_mean_std.mean) / np.sqrt(self.obs_mean_std.var + 1e-12)
    
    def _build_wrapped_envs(self, seed=None):
        self._wrapped_envs = []
        n_env = self.n_env
        assert n_env > 0 and type(n_env) is int
        for _ in range(n_env):
            seed = random.randint(0,65535)
            wrapped_env = make_gym_env(self.env_name)
            set_env_seed(wrapped_env, seed)
            self._wrapped_envs.append(wrapped_env)
        self.observation_space = self._wrapped_envs[0].observation_space
        ub = np.ones(self._wrapped_envs[0].action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    @property
    def wrapped_env(self):
        return self._wrapped_envs

    @property
    def horizon(self):
        return self.max_length

    def reset(self):
        self.cur_step_id = 0
        obs = []
        for wrapped_env in self._wrapped_envs:
            obs.append(wrapped_env.reset())
        obs = np.array(obs, dtype=np.float)
        if self.normalize:
            self.obs_mean_std.update(obs)
            obs = self._normalize_obs(obs)
        return obs

    def step(self, action):
        assert len(action) == self.n_env
        self.cur_step_id = self.cur_step_id + 1
        obs = []
        reward = []
        done = []
        info = {}
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        if len(action.shape) == len(self.action_shape):
            action = np.array([action] * self.n_env)

        for i in range(self.n_env):
            wrapped_env = self._wrapped_envs[i]
            a = action[i]
            s, r, d, _info = wrapped_env.step(a)
            obs.append(s)
            reward.append([r*self.reward_scale])
            done.append([d])
            for key in _info:
                if key in info:
                    info[key].append([_info[key]])
                else:
                    info[key]=[[_info[key]]]

        obs = np.array(obs, dtype=np.float)
        if self.normalize:
            self.obs_mean_std.update(obs)
            obs = self._normalize_obs(obs)
        reward = np.array(reward, dtype=np.float)
        done = np.array(done, dtype=np.float)
        if self.cur_step_id >= self.max_length:
            done = np.ones((self.n_env, 1), dtype=np.float)
        return obs, reward, done, info

    def render(self, which_env=0, *args, **kwargs):
        return self._wrapped_envs[which_env].render(*args, **kwargs)

    def terminate(self):
        for wrapped_env in self._wrapped_envs:
            if hasattr(wrapped_env, "terminate"):
                wrapped_env.terminate()

    def __str__(self):
        env_str = type(self).__name__
        env_str += "(n_env: %d, ["%self.n_env
        wrapped_env_str_list = []
        for wrapped_env in self._wrapped_envs:
            wrapped_env_str_list.append('{}'.format(wrapped_env)) 
        env_str += ', '.join(wrapped_env_str_list)
        env_str += '])'
        return env_str