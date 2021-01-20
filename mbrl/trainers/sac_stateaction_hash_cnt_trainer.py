
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.sac_hash_cnt_trainer import HashCntSACTrainer


class StateActionHashCntSACTrainer(HashCntSACTrainer):
    def __init__(self, **kwargs):
        HashCntSACTrainer.__init__(self, **kwargs)

    def log_mean_max_min_std(self, name, log_data):
        diagnostics = OrderedDict()
        diagnostics[name + 'Max'] = np.max(log_data)
        diagnostics[name + 'Mean'] = np.mean(log_data)
        diagnostics[name + 'Min'] = np.min(log_data)
        diagnostics[name + 'Std'] = np.std(log_data)
        return diagnostics

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        rewards = self.get_rewards_plus_bonus(torch.cat((obs,actions),1), rewards)
        """
        Alpha
        """
        new_action, policy_info = self.policy.action(
            obs, reparameterize=True, return_log_prob=True,
        )
        log_prob_new_action = policy_info['log_prob']
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_new_action + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha_if_not_automatic

        """
        QF 
        """
        # Make sure policy accounts for squashing functions like tanh correctly!
        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_value']
        next_action, next_policy_info = self.policy.action(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        log_prob_next_action = next_policy_info['log_prob']
        target_q_next_action = self.qf.value(next_obs, 
                                       next_action, 
                                       use_target_value=True, 
                                       return_info=False) - alpha * log_prob_next_action
    
        # target_q_next_action = self.qf.value(next_obs, 
        #                                 next_action, 
        #                                 use_target_value=True, 
        #                                 return_info=False)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_next_action
        qf_loss = ((q_value_ensemble - q_target.detach()) ** 2).mean()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            self.qf.update_target(self.soft_target_tau)
            
        """
        policy
        """
        q_new_action, _ = self.qf.value(obs, new_action, return_ensemble=False)
        policy_loss = (alpha*log_prob_new_action - q_new_action).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Compute some statistics for eval
        """

        average_entropy = -log_prob_new_action.mean()
        policy_q_loss = 0 - q_new_action.mean()

        diagnostics = OrderedDict()
        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = np.mean(ptu.get_numpy(policy_q_loss))
        diagnostics['Averaged Entropy'] = np.mean(ptu.get_numpy(average_entropy))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        # diagnostics['Terminals'] = np.mean(ptu.get_numpy(terminals))
        # diagnostics['Rewards'] = np.mean(ptu.get_numpy(rewards))
        if self.use_automatic_entropy_tuning:
            diagnostics['Alpha'] = alpha.item()
            diagnostics['Alpha Loss'] = alpha_loss.item()

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q_value_ensemble[0]),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q_value_ensemble[1]),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_prob_new_action),
            ))
            self.eval_statistics.update(diagnostics)
        self._n_train_steps_total += 1
        
        return diagnostics

class VisionHashSACTrainer(StateActionHashCntSACTrainer):
    def get_weight(self, states, actions):
        w = self.env.get_long_term_weight_batch(states, actions)
        return w

    def get_rewards_bonus(self, states, rewards):
        # states shape (batch_size, dim_state)
        # states torch 
        # rewards torch
        batch_phi = torch.matmul(states, torch.t(self.A))
        encode_batch_phi = torch.sign(batch_phi)
        zero = torch.zeros_like(encode_batch_phi)
        encode_batch_phi = torch.where(encode_batch_phi > -1, zero, encode_batch_phi)
        encode_batch_phi = encode_batch_phi.detach().cpu().numpy()
        reward_bonus_count = self._compute_batch_count(encode_batch_phi) # shape (batch_size)
        reward_bonus_count = torch.reshape(reward_bonus_count, rewards.shape)
        if self.cnt_with_sqrt:
            rewards_bonus = self.beta/torch.sqrt(reward_bonus_count+1).to(self.device)
        else:
            rewards_bonus = self.beta/(reward_bonus_count+1).to(self.device)
        return rewards_bonus

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        diagnostics = OrderedDict()
        diagnostics.update(self.log_mean_max_min_std('Reward Before', ptu.get_numpy(rewards)))

        # shaping rewards
        rewards_bonus = self.get_rewards_bonus(torch.cat((obs,actions),1), rewards)
        bonus_weight = self.get_weight(obs, actions)

        rewards = rewards + self.int_coeff * rewards_bonus * bonus_weight




        """
        Alpha
        """
        new_action, policy_info = self.policy.action(
            obs, reparameterize=True, return_log_prob=True,
        )
        log_prob_new_action = policy_info['log_prob']
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_new_action + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha_if_not_automatic

        """
        QF 
        """
        # Make sure policy accounts for squashing functions like tanh correctly!
        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_value']
        next_action, next_policy_info = self.policy.action(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        log_prob_next_action = next_policy_info['log_prob']
        target_q_next_action = self.qf.value(next_obs, 
                                       next_action, 
                                       use_target_value=True, 
                                       return_info=False) - alpha * log_prob_next_action

        # target_q_next_action = self.qf.value(next_obs, 
        #                                 next_action, 
        #                                 use_target_value=True, 
        #                                 return_info=False)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_next_action
        qf_loss = ((q_value_ensemble - q_target.detach()) ** 2).mean()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            self.qf.update_target(self.soft_target_tau)
            
        """
        policy
        """
        q_new_action, _ = self.qf.value(obs, new_action, return_ensemble=False)
        policy_loss = (alpha*log_prob_new_action - q_new_action).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Compute some statistics for eval
        """

        average_entropy = -log_prob_new_action.mean()
        policy_q_loss = 0 - q_new_action.mean()

        diagnostics.update(self.log_mean_max_min_std('Reward Bonus Real', ptu.get_numpy(rewards_bonus)))
        diagnostics.update(self.log_mean_max_min_std('Reward Weight', ptu.get_numpy(bonus_weight)))

        diagnostics.update(self.log_mean_max_min_std('Reward After', ptu.get_numpy(rewards)))

        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = np.mean(ptu.get_numpy(policy_q_loss))
        diagnostics['Averaged Entropy'] = np.mean(ptu.get_numpy(average_entropy))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        # diagnostics['Terminals'] = np.mean(ptu.get_numpy(terminals))
        # diagnostics['Rewards'] = np.mean(ptu.get_numpy(rewards))
        if self.use_automatic_entropy_tuning:
            diagnostics['Alpha'] = alpha.item()
            diagnostics['Alpha Loss'] = alpha_loss.item()

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q_value_ensemble[0]),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q_value_ensemble[1]),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_prob_new_action),
            ))
            self.eval_statistics.update(diagnostics)
        self._n_train_steps_total += 1
        
        return diagnostics