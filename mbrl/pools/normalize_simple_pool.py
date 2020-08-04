import numpy as np 

from mbrl.pools.simple_pool import SimplePool

class NormalizeSimplePool(SimplePool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False):
       SimplePool.__init__(self, env, max_size=1e6, compute_mean_std=False)

    def get_mean_std(self):
        obs_mean = np.mean(self.dataset['observations'], axis=0, keepdims=True)
        obs_std = np.std(self.dataset['observations'], axis=0, keepdims=True)

        ac_mean = np.mean(self.dataset['actions'], axis=0, keepdims=True)
        ac_std = np.std(self.dataset['actions'], axis=0, keepdims=True)
        
        obs_delta = self.dataset['next_observations'] - self.dataset['observations']

        obs_delta_mean = np.mean(obs_delta, axis=0, keepdims=True)
        obs_delta_std = np.std(obs_delta, axis=0, keepdims=True)

        mean_std_dict = {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "ac_mean": ac_mean,
            "ac_std": ac_std,
            "obs_delta_mean": obs_delta_mean,
            "obs_delta_std": obs_delta_std
        }

        return mean_std_dict