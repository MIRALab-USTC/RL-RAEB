import numpy as np
import warnings
import torch 

from collections import OrderedDict

from mbrl.utils.mean_std import RunningMeanStd
from mbrl.pools.base_pool import Pool
from mbrl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mbrl.collectors.utils import path_to_samples
from mbrl.pools.simple_pool import SimplePool

class SimplePoolWithTabular(SimplePool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False, beta=1.0):
        SimplePool.__init__(self, env, max_size, compute_mean_std)

        self.hash_table = {}
        for i in range(7):
            self.hash_table[i] = 0
        # parameter for simhash
        # we take identity function as g(s)
        #self.k = hash_k
        #self.d = env.observation_space.shape[0]
        #self.d = hash_d
        self.beta = beta

        #mu = torch.zeros((self.k, self.d))
        #std = torch.ones((self.k, self.d))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.A = torch.normal(mu, std).to(self.device)

    def add_samples(self, samples):
        for k in self.fields:
            v = samples[k]
            self._update_single_field(k,v)
        # update hash table
        self.update_hash_table(samples)
 
        stop = self._stop
        new_sample_size = len(samples[k])
        max_size = self.max_size
        self._stop = new_stop = (stop + new_sample_size) % max_size
        self._size = min(max_size, self._size + new_sample_size)
        for tag in self.unprocessed_stop:
            self.unprocessed_stop[tag] = new_stop
            unprocessed_size = self.unprocessed_size[tag] + new_sample_size
            if unprocessed_size > max_size:
                warnings.warn("unprocessed_size > max_size")
                self.unprocessed_size[tag] = max_size
            else:
                self.unprocessed_size[tag] = unprocessed_size
        return new_sample_size
    
    def get_state_block(self, state):
        x = state[3].item()
        y = state[2].item()

        if -1 < x < 1:
            x_block = 'low'
        elif 1 < x < 3:
            x_block = 'mid'
        elif 3 < x < 5:
            x_block = 'high'
        else:
            raise Exception

        if -1 < y < 1:
            y_block = 'left'
        elif 1 < y < 3:
            y_block = 'center'
        elif 3 < y < 5:
            y_block = 'right'
        else:
            raise Exception
        

        if x_block == 'low' and y_block == 'left':
            return 0
        elif x_block == 'low' and y_block == 'center':
            return 1
        elif x_block == 'low' and y_block == 'right':
            return 2
        elif x_block == 'mid' and y_block == 'right':
            return 3
        elif x_block == 'high' and y_block == 'right':
            return 4
        elif x_block == 'high' and y_block == 'center':
            return 5
        elif x_block == 'high' and y_block == 'left':
            return 6
        
    def update_hash_table(self, data):
        state = None
        if isinstance(data, dict):
            state = data['observations']
        else:
            state = data
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) > 1:
            self._update_batch(state)
        else:
            self._update(state)

    def _update(self, state):
        key = self.get_state_block(state)
        self.hash_table[key] += 1

    def _update_batch(self, state):
        # state shape (batch_size, dim_state)
        for i in range(state.shape[0]):
            key = self.get_state_block(state[i])
            self.hash_table[key] += 1

    def clear(self):
        # 清空pool
        self.hash_table = {}    

    def get_diagnostics(self):
        return self.hash_table