import numpy as np 

from mbrl.pools.simple_pool import SimplePool
from mbrl.pools.utils import random_batch_ensemble
from mbrl.pools.utils import _shuffer_and_random_batch_model

#from ipdb import set_trace

class NormalizeSimplePool(SimplePool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False):
       SimplePool.__init__(self, env, max_size=1e6, compute_mean_std=False)

    def get_mean_std(self):

        obs_mean = np.mean(self.dataset['observations'][:self._size], axis=0, keepdims=True)
        #set_trace()
        obs_std = np.std(self.dataset['observations'][:self._size], axis=0, keepdims=True)
        #set_trace()
        ac_mean = np.mean(self.dataset['actions'][:self._size], axis=0, keepdims=True)
        #set_trace()
        ac_std = np.std(self.dataset['actions'][:self._size], axis=0, keepdims=True)
        #set_trace()
        obs_delta = self.dataset['next_observations'][:self._size] - self.dataset['observations'][:self._size]

        obs_delta_mean = np.mean(obs_delta, axis=0, keepdims=True)
        obs_delta_std = np.std(obs_delta, axis=0, keepdims=True)
        #set_trace()
        mean_std_dict = {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "ac_mean": ac_mean,
            "ac_std": ac_std,
            "obs_delta_mean": obs_delta_mean,
            "obs_delta_std": obs_delta_std
        }
        return mean_std_dict
    
    def sample_ensemble_batch(self, batch_size, ensemble_size, keys=None):
        keys = self._check_keys(keys)
        return random_batch_ensemble(self.dataset, batch_size, self._size, ensemble_size, keys)

    def shuffer_and_random_batch_model(self, batch_size, ensemble_size, keys=None):
        keys = self._check_keys(keys)
        for batch in _shuffer_and_random_batch_model(self.dataset, batch_size, self._size, ensemble_size, keys):
            yield batch 