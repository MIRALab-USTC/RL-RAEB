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
from mbrl.trainers.sac import SACMaxEnt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SACPlanningTrainerStateEntropy(SACTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            model,

            policy_replay_size=int(1e7),
            policy_active_updates=1,
            policy_batch_size = 4096,
            policy_n_hidden = 256,
            alpha = 0.02,

            intrinsic_coeff=0.1,
            discount=0.99,
            reward_scale=1.0,

            model_lr=1e-3,
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
        SACTrainer.__init__(
            self,
            env,
            policy,
            qf,    
            discount=0.99,
            reward_scale=1.0,
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
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
        
        # 传入的model 是已经 normalizer设置过的
        self.model = model 
        self.intrinsic_coeff = intrinsic_coeff
        self.model_lr = model_lr
        self.training_noise_stdev = training_noise_stdev
        self.grad_clip = grad_clip

        self.model_optimizer = optimizer_class(
            self.model.parameters(),
            lr=model_lr,
        )        
        self._need_to_update_model = True

        d_state = self.env.observation_space.shape[0]
        d_action = self.env.action_space.shape[0]
        self.agent = SACMaxEnt(d_state, d_action, policy_replay_size, policy_batch_size, policy_active_updates, policy_n_hidden, discount, alpha, policy_lr, soft_target_tau)
        self.agent.to(device)

    def reward_function_novelty(self, obs, actions, next_obs):
        # resieze to  (ensemble_size, batch size, dim_state)
        # output: (ensemble_size, batch size, dim_state)


        obs =  obs.repeat(self.model.ensemble_size, 1, 1)
        actions = actions.repeat(self.model.ensemble_size, 1, 1)
        next_predicted_obs_mean, var = self.model(obs, actions)

        if self.model.ensemble_size == 1:
            next_predicted_obs_mean = torch.squeeze(next_predicted_obs_mean)
            var = torch.squeeze(var)
            p = Normal(next_predicted_obs_mean, torch.sqrt(var))

            # p(s^{\prime}|s,a) = \PI_{i=1}^{n} p(s^{\prime}_i)
            rewards_int = - self.intrinsic_coeff * torch.sum(p.log_prob(next_obs), axis=1, keepdim=True)
            return rewards_int
        else:
            raise NotImplementedError


    def train_model_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        diagnostics = OrderedDict()

        """
        model
        """
        model_loss = self.train_model_from_batch(obs, actions, next_obs - obs)
        diagnostics['Model Loss'] = np.mean(ptu.get_numpy(model_loss))
        if self._need_to_update_model:
            self._need_to_update_model = False
            self.eval_statistics.update(diagnostics)
        return diagnostics

    def train_from_torch_batch(self, batch):
        # train policy / value 
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        diagnostics = OrderedDict()
        # shaping reward
        rewards_int = self.reward_function_novelty(obs, actions, next_obs)
        rewards += rewards_int

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

        diagnostics['Reward Intrinsic'] = np.mean(ptu.get_numpy(rewards_int))
        diagnostics['Rewards'] = np.mean(ptu.get_numpy(rewards))
        
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

    def train_model_from_batch(self, states, actions, state_deltas):
        self.model_optimizer.zero_grad()

        model_loss = self.model.loss(states, actions, state_deltas, training_noise_stdev=self.training_noise_stdev)
        print(f"model_loss: {model_loss.item()}")
        model_loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
        self.model_optimizer.step()
        return model_loss

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self._need_to_update_model = True