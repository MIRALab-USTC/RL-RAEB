from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.base_trainer import BatchTorchTrainer
from mbrl.trainers.nnp_trainer import NNPTrainer
from mbrl.utils.logger import logger

class NNPTrainerMH(NNPTrainer):
    def compute_average_q_and_bonus(self, obs, use_target_value=False):
        batch_size = obs.shape[0]
        _, x_info = self.policy.action(
            obs, return_distribution=True, without_sampling=True,
            )
        actions = x_info['pretanh_actions']
        actions = torch.tanh(actions)
        x = actions.reshape(-1, self.action_size)
        probability = x_info['pretanh_probability']

        sample_number = actions.shape[-2]
        new_obs = obs.repeat(1,sample_number).reshape(batch_size*sample_number, -1)

        q = self.qf.value(new_obs, x, return_info=False, use_target_value=use_target_value)
        average_q = q.reshape(batch_size, -1, 1).mean(1)

        if 'phi' in self.bonus_type:
            x1 = x.reshape(batch_size, 1,sample_number, self.action_size)
            x2 = x.reshape(batch_size, sample_number,1, self.action_size)
            distance_x1_x2 = (x1 - x2)**2
            p1 = probability.reshape(batch_size, 1,sample_number, 1)
            p2 = probability.reshape(batch_size, sample_number,1, 1)
            weight_matrix = p2 * p1
            if self.bonus_type == 'phi_power':
                part1 = ((distance_x1_x2+1e-6) ** self.exponent).sum(dim=-1, keepdim=True)
            elif self.bonus_type == 'phi_log':
                part1 = torch.log(distance_x1_x2+1e-6).sum(dim=-1, keepdim=True)

            part1 = (part1 * weight_matrix).sum(dim=[-2,-3])
            part2 = self.torch_phi_f(actions).sum(dim=-1, keepdim=True)
            weight_vector = probability.reshape(batch_size, sample_number, 1)
            part2 = (part2 * weight_vector).sum(1)
            bonus = part1 + self.expectation_yy - 2*part2
            return average_q, bonus