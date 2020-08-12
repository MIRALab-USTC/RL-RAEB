from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions.normal import Normal

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.sac_trainer_surprise_based import SurpriseBasedSACTrainer


class SurpriseBasedSACTrainerVirtualLoss(SurpriseBasedSACTrainer):

    def __init__(
            self,
            env,
            policy,
            qf,
            model,
            virtual_pool,

            virtual_reward_decay=0.999,
            intrinsic_coeff=0.1,
            discount=0.99,
            reward_scale=1.0,

            model_lr=1e-6,
            training_noise_stdev=0,
            grad_clip=5,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class='Adam',

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            alpha_if_not_automatic=1e-2,
            use_automatic_entropy_tuning=True,
            init_log_alpha=0,
            target_entropy=None
    ):
        SurpriseBasedSACTrainer.__init__(
            self,
            env,
            policy,
            qf,
            model,
            intrinsic_coeff=0.1,
            discount=0.99,
            reward_scale=1.0,

            model_lr=1e-6,
            training_noise_stdev=0,
            grad_clip=5,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class='Adam',

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            alpha_if_not_automatic=1e-2,
            use_automatic_entropy_tuning=True,
            init_log_alpha=0,
            target_entropy=None
        )
        self.virtual_pool = virtual_pool
        self.virtual_reward_decay = virtual_reward_decay

    def reward_function_novelty(self, obs, actions, next_obs):
        # resieze to  (ensemble_size, batch size, dim_state)
        # output: (ensemble_size, batch size, dim_state)
        
        # virtual loss
        reward_virtual_count_obs = self.virtual_pool.compute_virtual_loss(obs)
        reward_virtual_count_next_obs = self.virtual_pool.compute_virtual_loss(next_obs)
        reward_virtual_count = reward_virtual_count_obs + reward_virtual_count_next_obs
        reward_virtual_count = reward_virtual_count.unsqueeze(1)
        reward_decay = (self.virtual_reward_decay ** reward_virtual_count).to(obs.device)
        #virtual_reward_loss = torch.sqrt(reward_virtual_count).to(obs.device)

        print(f"reward_virtual_count: {reward_virtual_count}")
        obs =  obs.repeat(self.model.ensemble_size, 1, 1)
        actions = actions.repeat(self.model.ensemble_size, 1, 1)
        next_predicted_obs_mean, var = self.model(obs, actions)

        if self.model.ensemble_size == 1:
            next_predicted_obs_mean = torch.squeeze(next_predicted_obs_mean)  # shape (batch_size, dim_state)
            var = torch.squeeze(var)
            p = Normal(next_predicted_obs_mean, torch.sqrt(var))


            # p(s^{\prime}|s,a) = \PI_{i=1}^{n} p(s^{\prime}_i)
            rewards_int = - self.intrinsic_coeff * torch.sum(p.log_prob(next_obs), axis=1, keepdim=True)
            #rewards_int = rewards_int - virtual_reward_loss
            rewards_int = rewards_int * reward_decay
            return rewards_int
        else:
            raise NotImplementedError
