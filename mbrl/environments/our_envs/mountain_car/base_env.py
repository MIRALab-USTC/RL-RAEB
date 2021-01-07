import abc
from gym.utils import seeding


class BaseEnv(object, metaclass=abc.ABCMeta):
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    action_space = None
    observation_space = None

    @abc.abstractclassmethod 
    def name(cls):
        pass

    def __init__(self, seed):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def env_info(self):
        """
        return a dict or None
        """
        return None

