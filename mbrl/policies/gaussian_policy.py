from mbrl.policies.base_policy import RandomPolicy
from mbrl.torch_modules.policies import MeanLogstdGaussianPolicyModule, TanhPolicyModule
from mbrl.utils.logger import logger
import torch
from torch import nn
from torch.distributions import Normal

class GaussianPolicy(nn.Module, RandomPolicy):
    def __init__( self, 
                  env, 
                  obs_processor=None,
                  deterministic=False,
                  tanh_action=True,
                  policy_name='gaussian_policy',
                  **mlp_kwargs):
        nn.Module.__init__(self)
        RandomPolicy.__init__(self, env, obs_processor, deterministic)
        assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1
        gaussian = MeanLogstdGaussianPolicyModule( self.processed_obs_shape[0], 
                                                   self.action_shape[0],
                                                   policy_name,
                                                   **mlp_kwargs)
        self.tanh_action = tanh_action

        if tanh_action:
            self.module = TanhPolicyModule(gaussian)
        else:
            self.module = gaussian
        
    def _action( self, 
                 obs, 
                 return_info=True,
                 return_log_prob=False,
                 **kwargs
                ):
        return self.module(obs, 
                            deterministic=self._deterministic, 
                            return_info=return_info, 
                            return_log_prob=return_log_prob,
                            **kwargs)

    def _log_prob(self, obs, action):
        return self.module.log_prob(obs, action)

    def save(self, save_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir)
    
    def load(self, load_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.load(save_dir)

    def get_snapshot(self):
        return self.module.get_snapshot()

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class SimpleGaussianPolicy(nn.Module):
    def __init__(self, env, n_hidden):
        super().__init__()
        d_state = env.observation_space.shape[0]
        d_action = env.action_space.shape[0]

        one = nn.Linear(d_state, n_hidden)
        self.init_weights(one)
        two = nn.Linear(n_hidden, n_hidden)
        self.init_weights(two)
        three = nn.Linear(n_hidden, 2 * d_action)
        self.init_weights(three)

        self.layers = nn.Sequential(one,
                                    nn.LeakyReLU(),
                                    two,
                                    nn.LeakyReLU(),
                                    three)

    def forward(self, state):
        y = self.layers(state)
        mu, log_std = torch.split(y, y.size(1) // 2, dim=1)

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        pi = normal.rsample()           # with re-parameterization
        logp_pi = normal.log_prob(pi).sum(dim=1, keepdim=True)

        # bounds
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        logp_pi -= torch.sum(torch.log(torch.clamp(1 - pi.pow(2), min=0, max=1) + EPS), dim=1, keepdim=True)

        return pi, logp_pi, mu, log_std

    def init_weights(self, layer):
        nn.init.orthogonal_(layer.weight)
        nn.init.constant_(layer.bias, 0)