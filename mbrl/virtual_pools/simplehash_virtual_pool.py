import torch
import numpy as np

from mbrl.virtual_pools.base_virtual_pool import VirtualPool

# Implementing SimHash from paper: Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning

class SimpleVirtualPool(VirtualPool):
    def __init__(self, env, d=4, k=3):
        self.hash_table = {}
        self._env = env

        # parameter for simhash
        self.k = k
        self.d = self._env.observation_space.shape[0]

        mu = torch.zeros((self.k, self.d))
        std = torch.ones((self.k, self.d))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.A = torch.normal(mu, std).to(self.device)

        self.states = []

    def add_samples(self, states):
        self.states.append(states)

    def update_hash_table(self, data):
        state = None
        if isinstance(data, dict):
            state = data['observations']
            #next_state = data['next_observations']
        else:
            state = data
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        #next_state = torch.FloatTensor(next_state).to(self.device)
        if len(state.shape) > 1:
            self._update_batch(state)
            #self._update_batch(next_state)
        else:
            self._update(state)
            #self._update(next_state)

    def _update(self, state):
        phi = torch.matmul(self.A, state)
        encode_phi = torch.sign(phi)
        if encode_phi in self.hash_table.keys():
            self.hash_table[encode_phi] += 1
        else:
            self.hash_table[encode_phi] = 1

    def _update_batch(self, state):
        # state shape (batch_size, dim_state)
        phi = torch.matmul(state, torch.t(self.A))
        encode_phi = torch.sign(phi)  # shape (batch_size, k)
        for i in range(encode_phi.shape[0]):
            if encode_phi[i] in self.hash_table.keys():
                self.hash_table[encode_phi[i]] += 1
            else:
                self.hash_table[encode_phi[i]] = 1

    def compute_virtual_loss(self, states):
        # states shape (batch_size, dim_state)
        batch_phi = torch.matmul(states, torch.t(self.A))
        encode_batch_phi = torch.sign(batch_phi)
        reward_decay_count = self._compute_batch(encode_batch_phi)
        
        return reward_decay_count  # shape (batch_size)

    def _compute_batch(self, encode_batch_phi):
        virtual_count = torch.zeros(encode_batch_phi.shape[0])
        for i in range(encode_batch_phi.shape[0]):
            if encode_batch_phi[i] in self.hash_table.keys():
                virtual_count[i] = self.hash_table[encode_batch_phi[i]]
        return virtual_count

    def clear(self):
        # 清空pool
        self.hash_table = {}    

    def get_diagnostics(self):
        return self.hash_table