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


class RNDSACTrainer(SACTrainer):
    def __init__(
        self,
        env,
        policy,
        qf,
        random_model,
        random_target_model,
        intrinsic_coeff,
        model_lr,
        **sac_kwargs
    ):
        SACTrainer.__init__(self, env, policy, qf, **sac_kwargs)
        self.random_model = random_model
        self.random_target_model = random_target_model
        self.intrinsic_coeff = intrinsic_coeff
        self.model_lr = model_lr

        self.model_optimizer = self.optimizer_class(
            self.random_model.parameters(),
            lr=model_lr,
        )        
        self._need_to_update_model = True
        self._need_to_update_int_reward = True

    def log_mean_max_min_std(self, name, log_data):
        diagnostics = OrderedDict()
        diagnostics[name + 'Max'] = np.max(log_data)
        diagnostics[name + 'Mean'] = np.mean(log_data)
        diagnostics[name + 'Min'] = np.min(log_data)
        diagnostics[name + 'Std'] = np.std(log_data)
        return diagnostics
    
    def train_model_from_torch_batch(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        diagnostics = OrderedDict()

        """
        data_preprocess
        """
        if self.random_model.input_mode == "state":
            x = obs
        elif self.random_model.input_mode == "state_action":
            x = torch.cat((obs, actions), 1)
        
        with torch.no_grad():
            y_target = self.random_target_model(x) 
        y = self.random_model(x) 

        # model loss
        model_loss = torch.mean((y-y_target)**2, dim=1)
        model_loss = torch.mean(model_loss)

        # optimize model
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        diagnostics['Model Loss'] = np.mean(ptu.get_numpy(model_loss))
        if self._need_to_update_model:
            self._need_to_update_model = False
            self.eval_statistics.update(diagnostics)
        return diagnostics

    def reward_function_novelty(self, obs, actions, next_obs):
        diagnostics = OrderedDict()
        if self.random_model.input_mode == "state":
            x = obs 
        elif self.random_model.input_mode == "state_action":
            x = torch.cat((obs,actions), 1)

        y_target = self.random_target_model(x)
        y = self.random_model(x)
        reward_int = torch.sum((y-y_target)**2, dim=1, keepdim=True)

        # log
        diagnostics = self.log_mean_max_min_std('reward_int', ptu.get_numpy(reward_int))
        
        if self._need_to_update_int_reward:
            self._need_to_update_int_reward = False
            self.eval_statistics.update(diagnostics)
        return reward_int

    def train_from_torch_batch(self, batch):
        # train policy / value 
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        diagnostics = OrderedDict()
        # shaping reward
        # Todo
        # add reward_int min max log
        rewards_int = self.reward_function_novelty(obs, actions, next_obs)
        eta = self.get_eta(rewards_int)
        rewards = rewards + eta * rewards_int
        diagnostics['eta'] = eta
        diagnostics['discount'] = self.discount

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
        #                                next_action, 
        #                                use_target_value=True, 
        #                                return_info=False)
                                        
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

        diagnostics.update(self.log_mean_max_min_std('Reward Intrinsic Real', ptu.get_numpy(rewards_int)))
        diagnostics.update(self.log_mean_max_min_std('Rewards Shaping', ptu.get_numpy(rewards)))
        
        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = np.mean(ptu.get_numpy(policy_q_loss))
        diagnostics['Averaged Entropy'] = np.mean(ptu.get_numpy(average_entropy))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        # diagnostics['Terminals'] = np.mean(ptu.get_numpy(terminals))
        
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

    def get_eta(self, rewards_int):
        eta = self.intrinsic_coeff / max(1, np.mean(ptu.get_numpy(rewards_int)))
        return eta

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self._need_to_update_model = True
        self._need_to_update_int_reward = True


class VisionRNDSACTrainer(RNDSACTrainer):
    def get_weight(self, states, actions):
        w = self.env.get_long_term_weight_batch(states, actions)
        return w

    def train_from_torch_batch(self, batch):
        # train policy / value 
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        diagnostics = OrderedDict()
        # shaping reward
        # Todo
        # add reward_int min max log
        rewards_int = self.reward_function_novelty(obs, actions, next_obs)
        rewards_int_weight = self.get_weight(obs, actions)
        eta = self.get_eta(rewards_int)
        rewards = rewards + eta * rewards_int
        diagnostics['eta'] = eta
        diagnostics['discount'] = self.discount

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

        #target_q_next_action = self.qf.value(next_obs, 
        #                                next_action, 
        #                                use_target_value=True, 
        #                                return_info=False)
                                            
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

        diagnostics.update(self.log_mean_max_min_std('Reward Intrinsic Real', ptu.get_numpy(rewards_int)))
        diagnostics.update(self.log_mean_max_min_std('Reward Int Weight', ptu.get_numpy(rewards_int_weight)))
        diagnostics.update(self.log_mean_max_min_std('Rewards Shaping', ptu.get_numpy(rewards)))
        
        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = np.mean(ptu.get_numpy(policy_q_loss))
        diagnostics['Averaged Entropy'] = np.mean(ptu.get_numpy(average_entropy))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        # diagnostics['Terminals'] = np.mean(ptu.get_numpy(terminals))
        
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