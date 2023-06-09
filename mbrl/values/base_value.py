import abc
import mbrl.torch_modules.utils as ptu

class Value(object, metaclass=abc.ABCMeta):
    def __init__(self, env, obs_processor=None):
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        if obs_processor is None:
            self.processed_obs_shape = self.observation_shape
        else:
            self.processed_obs_shape = obs_processor.output_shape
        self._obs_processor = obs_processor

    def save(self, save_dir=None):
        raise NotImplementedError
     
    def load(self, load_dir=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}


class StateValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _value(self, obs, return_info=True):
        pass

    def value(self, obs, return_info=True, **kwargs):
        if self._obs_processor is not None:
            obs = self._obs_processor.process(obs)
        return self._value(obs, return_info=return_info, **kwargs)

    def value_np(self, obs, return_info=True, **kwargs):
        obs = ptu.from_numpy(obs)
        if return_info:
            value, info = self.value(obs, return_info=return_info, **kwargs)
            value = ptu.get_numpy(value)
            info = ptu.torch_to_np_info(info)
            return value, info
        else:
            return ptu.get_numpy(self.value(obs, return_info=return_info, **kwargs))


class QValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _value(self, obs, action, return_info=True):
        pass

    def value(self, obs, action, return_info=True, **kwargs):
        if self._obs_processor is not None:
            obs = self._obs_processor.process(obs)
        return self._value(obs, action, return_info=return_info, **kwargs)

    def value_np(self, obs, action, return_info=True, **kwargs):
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        if return_info:
            value, info = self.value(obs, action, return_info=return_info, **kwargs)
            value = ptu.get_numpy(value)
            info = ptu.torch_to_np_info(info)
            return value, info
        else:
            return ptu.get_numpy(self.value(obs, action, return_info=return_info, **kwargs))
