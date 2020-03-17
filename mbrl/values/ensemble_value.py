import torch
from torch import nn
import numpy as np
from mbrl.torch_modules.mlp import MLP
import mbrl.torch_modules.utils as ptu
from mbrl.utils.logger import logger
from mbrl.values.base_value import StateValue, QValue

class EnsembleStateValue(nn.Module, StateValue):
    def __init__( self, 
                  env, 
                  ensemble_size=2, 
                  obs_processor=None,
                  with_target_value=True,
                  value_name='ensemble_state_value',
                  **mlp_kwargs
                ):
        nn.Module.__init__(self)
        StateValue.__init__(self, env, obs_processor)
        self.with_target_value = with_target_value
        self.ensemble_size = ensemble_size
        assert len(self.processed_obs_shape) == 1
        self.module = MLP( self.processed_obs_shape[0], 
                           1,
                           ensemble_size=ensemble_size,
                           module_name=value_name,
                           **mlp_kwargs)
        if with_target_value:
            self.target_module = MLP( self.processed_obs_shape[0], 
                               1,
                               ensemble_size=ensemble_size,
                               module_name=value_name,
                               **mlp_kwargs)
            self.update_target(1)
    
    def _value(self, obs, return_info=False, use_target_value=False, mode='min', return_ensemble=False):

        if self.ensemble_size is not None:
            if use_target_value:
                ensemble_value = self.target_module(obs)
            else:
                ensemble_value = self.module(obs)

            if mode == 'min':
                value = torch.min(ensemble_value, dim=0)[0]
            elif mode == 'mean':
                value = torch.mean(ensemble_value, dim=0)[0]
            elif mode == 'max':
                value = torch.max(ensemble_value, dim=0)[0]
            elif mode == 'sample':
                index = np.random.randint(self.ensemble_size)
                value = ensemble_value[index]
            else:
                raise NotImplementedError

        else:
            if use_target_value:
                value = self.target_module(obs)
            else:
                value = self.module(obs)

        if return_info:
            info = {}
            if return_ensemble and self.ensemble_size is not None:
                info['ensemble_value'] = ensemble_value
            return value, info
        else:
            return value

    def update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.module, self.target_module)
        else:
            ptu.soft_update_from_to(self.module, self.target_module, tau)

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


class EnsembleQValue(nn.Module, QValue):
    def __init__( self, 
                  env, 
                  ensemble_size=2, 
                  obs_processor=None,
                  with_target_value=True,
                  value_name='ensemble_state_value',
                  **mlp_kwargs
                ):
        nn.Module.__init__(self)
        QValue.__init__(self, env, obs_processor)
        self.with_target_value = with_target_value
        self.ensemble_size = ensemble_size
        assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1
        self.module = MLP( self.processed_obs_shape[0] + self.action_shape[0], 
                           1,
                           ensemble_size=ensemble_size,
                           module_name=value_name,
                           **mlp_kwargs)
        if with_target_value:
            self.target_module = MLP( self.processed_obs_shape[0] + self.action_shape[0], 
                               1,
                               ensemble_size=ensemble_size,
                               module_name=value_name,
                               **mlp_kwargs)
            self.update_target(1)
    
    def _value(self, obs, action, return_info=False, use_target_value=False, mode='min', return_ensemble=False):
        input_tensor = torch.cat([obs, action], dim=-1)

        if self.ensemble_size is not None:
            if use_target_value:
                ensemble_value = self.target_module(input_tensor)
            else:
                ensemble_value = self.module(input_tensor)

            if mode == 'min':
                value = torch.min(ensemble_value, dim=0)[0]
            elif mode == 'mean':
                value = torch.mean(ensemble_value, dim=0)[0]
            elif mode == 'max':
                value = torch.max(ensemble_value, dim=0)[0]
            elif mode == 'sample':
                index = np.random.randint(self.ensemble_size)
                value = ensemble_value[index]
            else:
                raise NotImplementedError

        else:
            if use_target_value:
                value = self.target_module(input_tensor)
            else:
                value = self.module(input_tensor)

        if return_info:
            info = {}
            if return_ensemble and self.ensemble_size is not None:
                info['ensemble_value'] = ensemble_value
            return value, info
        else:
            return value

    def update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.module, self.target_module)
        else:
            ptu.soft_update_from_to(self.module, self.target_module, tau)

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