from mbrl.policies.base_policy import RandomPolicy
from mbrl.torch_modules.policies import MultiHeadPolicyModule, TanhPolicyModule
from torch import nn
import torch

class MultiHeadPolicy(nn.Module, RandomPolicy):
    def __init__( self, 
                  env, 
                  obs_processor=None,
                  deterministic=False,
                  number_of_heads=16,
                  learn_probability=False,
                  tanh_action=True,
                  with_expectation=True,
                  policy_name='multi_head_policy',
                  **mlp_kwargs):

        nn.Module.__init__(self)
        RandomPolicy.__init__(self, env, obs_processor, deterministic)
        assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1
        multi_head = MultiHeadPolicyModule(self.processed_obs_shape[0], 
                                           self.action_shape[0],
                                           number_of_heads,
                                           learn_probability,
                                           with_expectation,
                                           policy_name,
                                           **mlp_kwargs)
        self.tanh_action = tanh_action
        
        if tanh_action:
            self.module = TanhPolicyModule(multi_head, discrete=False)
        else:
            self.module = multi_head
    
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


