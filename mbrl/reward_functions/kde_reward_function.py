import torch
import numpy as np
import time 

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA 
from sklearn.model_selection import GridSearchCV

from mbrl.reward_functions.base_reward_function import BaseRewardFn
# Implementing KDE from paper: Provably Efficient Maximum Entropy Exploration

class KDERewardFn(BaseRewardFn):
    def __init__(self, env, n_components=32, eps=.001):

        self.eps = eps
        self.kde = None
        self.pca = None
        self._env = env
        self.env_name = self._env.env_name
        self.n_components = n_components

    def update(self, data):
        if data is not None:
            data = np.array(data)
            if 'ant' in self.env_name or 'human' in self.env_name:
                if self.n_components is not None:
                    start = time.time()
                    self.pca = PCA(n_components=n_components, whiten=False)
                    PCA_time = time.time() - start
                    print(f"PCA_time: {PCA_time}")
                    data = self.pca.fit_transform(data)
                    
            self.kde = self.fit_distribution(data)
    
    def fit_distribution(self, data):
        start = time.time()
        kde = KernelDensity(bandwidth=.1, kernel='epanechnikov').fit(data)
        kde_time = time.time() - start
        print(f"kde_time: {kde_time}")
        return kde

    def reward(self, x):
        print(f"x_shape: {x.shape}")
        if self.kde is None:
            return torch.zeros(x.shape[0], 1)
        
        prob_x = self.get_prob(x)
        return 1/(prob_x + self.eps)

    def get_prob(self, x):
        if self.pca is not None:
            x = self.pca.transform(x)
        return np.exp(self.kde.score(x))
