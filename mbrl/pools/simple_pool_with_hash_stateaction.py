import numpy as np
import warnings
import torch 

from collections import OrderedDict

from mbrl.utils.mean_std import RunningMeanStd
from mbrl.pools.base_pool import Pool
from mbrl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mbrl.collectors.utils import path_to_samples
from mbrl.pools.simple_pool_with_hash import SimplePoolWithHash

class SimplePoolWithHashStateAction(SimplePoolWithHash):
    def __init__(self, **pool_kwargs):
        SimplePoolWithHash.__init__(self, **pool_kwargs)
        self.d = pool_kwargs['env'].observation_space.shape[0] + pool_kwargs['env'].action_space.shape[0]
        mu = torch.zeros((self.k, self.d))
        std = torch.ones((self.k, self.d))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.normal(mu, std).to(self.device)
    
    def update_hash_table(self, data):
        state = None
        action = None
        if isinstance(data, dict):
            state = data['observations']
            action = data['actions']
        else:
            state = data
        print(f"state_shape: {state.shape}")
        print(f"action_shape: {action.shape}")
        state = np.concatenate((state,action), axis=1)
        print(f"state_action_shape: {state.shape}")
        
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) > 1:
            self._update_batch(state)
        else:
            self._update(state)

