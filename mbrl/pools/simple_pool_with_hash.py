import numpy as np
import warnings
import torch 

from collections import OrderedDict

from mbrl.utils.mean_std import RunningMeanStd
from mbrl.pools.base_pool import Pool
from mbrl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mbrl.collectors.utils import path_to_samples
from mbrl.pools.simple_pool import SimplePool

class SimplePoolWithHash(SimplePool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False, hash_d=4, hash_k=3, beta=1.0):
        SimplePool.__init__(self, env, max_size, compute_mean_std)

        self.hash_table = {}

        # parameter for simhash
        # we take identity function as g(s)
        self.k = hash_k
        self.d = env.observation_space.shape[0]
        #self.d = hash_d
        self.beta = beta

        mu = torch.zeros((self.k, self.d))
        std = torch.ones((self.k, self.d))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.normal(mu, std).to(self.device)

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
        phi = torch.matmul(self.A, state)

        encode_phi = torch.sign(phi)

        zero = torch.zeros_like(encode_phi)
        encode_phi = torch.where(encode_phi > -1, zero, encode_phi)
        encode_phi = encode_phi.detach().cpu().numpy()
        if encode_phi.tostring() in self.hash_table.keys():
            self.hash_table[encode_phi.tostring()] += 1
        else:
            self.hash_table[encode_phi.tostring()] = 1

    def _update_batch(self, state):
        # state shape (batch_size, dim_state)
        phi = torch.matmul(state, torch.t(self.A))

        encode_phi = torch.sign(phi)  # shape (batch_size, k)

        zero = torch.zeros_like(encode_phi)
        encode_phi = torch.where(encode_phi > -1, zero, encode_phi)
        encode_phi = encode_phi.detach().cpu().numpy()
        for i in range(encode_phi.shape[0]):
            if encode_phi[i].tostring() in self.hash_table.keys():
                self.hash_table[encode_phi[i].tostring()] += 1
            else:
                self.hash_table[encode_phi[i].tostring()] = 1



    def clear(self):
        # 清空pool
        self.hash_table = {}    

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])