from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions.normal import Normal

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.base_trainer import BatchTorchTrainer
from mbrl.trainers.sac_trainer import SACTrainer
from mbrl.trainers.sac_trainer_rnd import RNDSACTrainer

# # intrinsic reward 调整成rnd novelty difference, 加上alpha 参数，取1？
class RNDNovelDTrainer(RNDSACTrainer):
    def __init__(
        self,
        env,
        policy,
        qf,
        alpha,
        random_model,
        random_target_model,
        intrinsic_coeff,
        model_lr,
        **sac_kwargs
    ):
        RNDSACTrainer.__init__(
            self,
            env,
            policy,
            qf,
            random_model,
            random_target_model,
            intrinsic_coeff,
            model_lr,
            **sac_kwargs
        )
        self.alpha = alpha

    def reward_function_novelty(self, obs, actions, next_obs):
        diagnostics = OrderedDict()
        # cur state reward int
        if self.random_model.input_mode == "state":
            x = obs 
        elif self.random_model.input_mode == "state_action":
            x = torch.cat((obs,actions), 1)

        y_target = self.random_target_model(x)
        y = self.random_model(x)
        cur_state_reward_int = torch.sum((y-y_target)**2, dim=1, keepdim=True)

        # next state reward int 
        next_y_target = self.random_target_model(next_obs)
        next_y = self.random_model(next_obs)
        next_state_reward_int = torch.sum((next_y-next_y_target)**2, dim=1, keepdim=True)

        reward_int = next_state_reward_int - self.alpha*cur_state_reward_int
        zeros = torch.zeros_like(reward_int, dtype=reward_int.dtype)
        
        indexes_lower_zero = torch.where(reward_int.float()<0)
        reward_int[indexes_lower_zero] = zeros[indexes_lower_zero]

        # log
        diagnostics = OrderedDict()
        diagnostics.update(self.log_mean_max_min_std('reward_int', ptu.get_numpy(reward_int)))
        diagnostics.update(self.log_mean_max_min_std('cur state reward int', ptu.get_numpy(cur_state_reward_int)))
        diagnostics.update(self.log_mean_max_min_std('next state reward int', ptu.get_numpy(next_state_reward_int)))

        if self._need_to_update_int_reward:
            self._need_to_update_int_reward = False
            self.eval_statistics.update(diagnostics)
        return reward_int
