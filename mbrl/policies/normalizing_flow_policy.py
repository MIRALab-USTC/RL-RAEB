from mbrl.policies.base_policy import RandomPolicy
from mbrl.torch_modules.policies import NormalizingFlowPolicyModule, TanhPolicyModule
from torch import nn
import torch

class NormalizingFlowPolicy(nn.Module, RandomPolicy):
    def __init__( self, 
                  env, 
                  obs_processor=None,
                  deterministic=False,
                  n_blocks=4, 
                  n_hidden_per_block=3,
                  tanh_action=True,
                  policy_name='generator_policy',
                  **mlp_kwargs):

        nn.Module.__init__(self)
        RandomPolicy.__init__(self, env, obs_processor, deterministic)
        assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1
        nf = NormalizingFlowPolicyModule(self.processed_obs_shape[0], 
                                           self.action_shape[0],
                                           n_blocks,
                                           n_hidden_per_block,
                                           policy_name,
                                           **mlp_kwargs)
        self.tanh_action = tanh_action
        
        if tanh_action:
            self.module = TanhPolicyModule(nf, discrete=True)
        else:
            self.module = nf
    
    def _action( self, 
                 obs, 
                 return_info=True,
                 **kwargs
                ):
        return self.module(obs, 
                           deterministic=self._deterministic, 
                           return_info=return_info,
                           **kwargs)

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


