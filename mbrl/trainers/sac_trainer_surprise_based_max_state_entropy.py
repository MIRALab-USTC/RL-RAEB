from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.distributions.normal import Normal

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.sac_trainer_surprise_based import SurpriseBasedSACTrainer
from mbrl.environments.image_envs.imagination import Imagine

class SurpriseBasedSACTrainerMSE(SurpriseBasedSACTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,
            model,
            virtual_pool,

            virtual_reward_decay=0.999,
            virtual_bonus_beta=1.0,
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
        self.virtual_bonus_beta = virtual_bonus_beta

    def reward_function_novelty(self, obs, actions, next_obs):
        # resieze to  (ensemble_size, batch size, dim_state)
        # output: (ensemble_size, batch size, dim_state)
        diagnostics = OrderedDict()
        # virtual loss
        #reward_virtual_count_obs = self.virtual_pool.compute_virtual_loss(obs)
        reward_virtual_count = self.virtual_pool.compute_virtual_loss(next_obs)
        #reward_virtual_count = reward_virtual_count_obs + reward_virtual_count_next_obs
        reward_virtual_count = reward_virtual_count.unsqueeze(1)
        reward_virtual_count = torch.where(reward_virtual_count > 1000, torch.full_like(reward_virtual_count, 1000), reward_virtual_count)
        #reward_virtual = self.virtual_bonus_beta / torch.sqrt(1.0+reward_virtual_count)
        reward_virtual = self.virtual_reward_decay ** reward_virtual_count
        # reward_decay = (self.virtual_reward_decay ** reward_virtual_count).to(obs.device)
        #virtual_reward_loss = torch.sqrt(reward_virtual_count).to(obs.device)
        reward_virtual = reward_virtual.to(obs.device)
        diagnostics['state_entropy'] = np.mean(ptu.get_numpy(reward_virtual))

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
            diagnostics['model_int_reward'] = np.mean(ptu.get_numpy(rewards_int))
            rewards_int = rewards_int * reward_virtual
            diagnostics['rewards_int'] = np.mean(ptu.get_numpy(rewards_int))
            if self._need_to_update_int_reward:
                self._need_to_update_int_reward = False
                self.eval_statistics.update(diagnostics)

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
        states = states.repeat(self.model.ensemble_size,1,1)
        actions = actions.repeat(self.model.ensemble_size,1,1)
        state_deltas = state_deltas.repeat(self.model.ensemble_size,1,1)

        model_loss = self.model.loss(states, actions, state_deltas, training_noise_stdev=self.training_noise_stdev)
        model_loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
        self.model_optimizer.step()
        
        # update virtual pool
        state_distri_estimator = StateEstimator(self.policy, self.env, self.model, self.virtual_pool)
        self.virtual_pool = state_distri_estimator.update_virtual_pool()

        return model_loss


class StateEstimator:
    def __init__(
            self,
            policy,
            env,
            model,
            virtual_pool,

            n_actors=128,
            policy_horizon=50,
            num_init_states=20
        ):

        self.policy = policy
        self.env = env
        self.model = model 
        self.imagine_mdp = Imagine(model, n_actors, policy_horizon)
        self.virtual_pool = virtual_pool

        self.num_init_states = num_init_states

        self.init_states = None

    def update_virtual_pool(self):
        self.virtual_pool.clear()
        self.get_init_states()

        for i in range(self.num_init_states):
            init_state = self.init_states[i]
            self.imagine_mdp.update_init_state(init_state)
            # collect n_actors * episode data
            self.episode()
        
        return self.virtual_pool


    def get_init_states(self):
        if self.init_states is None:
            self.init_states = np.zeros((self.num_init_states, self.env.observation_space.shape[0]))
        for i in range(self.num_init_states):
            state = self.env.reset()
            self.init_states[i] = state # numpy array shape: (num_init_states, obs_shape)

    def episode(self):
        ep_length = 0
        done = False
        states = self.imagine_mdp.reset() # shape: (n_actors, obs_shape)
        while not done:
            with torch.no_grad():
                actions, _ = self.policy.action(
                    states, reparameterize=True, return_log_prob=False,
                )
            next_states, rewards, done, _ = self.imagine_mdp.step(actions)
            
            self.virtual_pool.update_hash_table(next_states)
            ep_length += 1

            states = next_states