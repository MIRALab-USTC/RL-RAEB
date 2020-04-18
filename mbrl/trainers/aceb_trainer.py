from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
from torch import nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.base_trainer import BatchTorchTrainer
from mbrl.utils.logger import logger

class ACEBTrainer(BatchTorchTrainer):
    def __init__(
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
            
            sample_number=2,

            bonus_type='phi_power',
            use_automatic_bonus_tuning=True,
            alpha_if_not_automatic=0,
            target_gaussian_std=0.15,

            end_target_gaussian_std=None,
            end_reducing_exploration=1000,

            plotter=None,
            render_eval_paths=False,
            
            **bonus_kwargs
    ):
        super().__init__()
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
        self.env = env
        self.policy = policy
        self.qf = qf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.action_size = np.prod(self.env.action_space.shape).item()
        self.bonus_type= bonus_type
        self.bonus_kwargs = bonus_kwargs
        self.sample_number = sample_number
        self.init_bonus_functions()

        self.alpha_if_not_automatic = alpha_if_not_automatic
        self.use_automatic_bonus_tuning = use_automatic_bonus_tuning
        if self.use_automatic_bonus_tuning:
            self.target_bonus = self.start_target_bonus = self.get_recommended_target(target_gaussian_std)  # heuristic value 
            if end_target_gaussian_std is None:
                self.end_target_bonus = self.start_target_bonus
            else:
                self.end_target_bonus = self.get_recommended_target(end_target_gaussian_std)  # heuristic value 

            logger.log("start target bonus: %f"%self.start_target_bonus)
            logger.log("end target bonus: %f"%self.end_target_bonus)
            self.end_reducing_exploration = end_reducing_exploration
            
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.policy_optimizer = optimizer_class(
            self.policy.module.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.module.parameters(),
            lr=qf_lr,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def init_bonus_functions(self):
        if self.bonus_type == 'entropy':
            return
        
        elif self.bonus_type == 'variance':
            return

        elif 'phi' in self.bonus_type:            
            n = self.sample_number
            weight_matrix = ptu.ones(n,n) - torch.diag(ptu.ones(n))
            self.weight_matrix = ( weight_matrix / (n * (n-1)) ).reshape(n,n,1,1)
            self.only_xx = self.bonus_kwargs.get('only_xx', False)

            if self.bonus_type == 'phi_power':
                self.exponent = e = self.bonus_kwargs.get('exponent', 0.25)
                self.expectation_yy = ( 2**(2*e+2) / (4*e+2) / (2*e+2) ) * self.action_size
                def phi_f(x):
                    a = 2*e+1
                    y = ( (1+x+1e-6)**a + (1-x+1e-6)**a ) / (a*2)
                    return y
                self.torch_phi_f = self.numpy_phi_f = phi_f

            if self.bonus_type == 'phi_log':
                self.expectation_yy = (2*math.log(2)-3) * self.action_size
                def numpy_phi_f(x):
                    y = (1-x)*np.log(1+1e-6-x) + (1+x)*np.log(1+1e-6+x) - 2
                    return y
                def torch_phi_f(x):
                    y = (1-x)*torch.log(1+1e-6-x) + (1+x)*torch.log(1+1e-6+x) - 2
                    return y
                self.numpy_phi_f = numpy_phi_f
                self.torch_phi_f = torch_phi_f
    
    def get_recommended_target(self, target_gaussian_std):
        action_size = self.action_size
        if self.bonus_type == 'entropy':
            return (0.5 + 0.5 * math.log(2 * math.pi) + math.log(target_gaussian_std) ) * action_size

        if self.bonus_type == 'variance':
            return (target_gaussian_std ** 2)  * action_size
        
        elif 'phi' in self.bonus_type:
            x1_samples = np.tanh(np.random.randn(2048, action_size) * target_gaussian_std) 
            x2_samples = np.tanh(np.random.randn(2048, action_size) * target_gaussian_std) 
            #x1_samples = np.random.rand(2048, action_size) * 2 - 1
            #x2_samples = np.random.rand(2048, action_size) * 2 - 1
            distance_x1_x2 = (x1_samples - x2_samples)**2

            if self.bonus_type == 'phi_power':
                part1 = np.sum((distance_x1_x2+1e-6) ** self.exponent, axis=-1)
            elif self.bonus_type == 'phi_log':
                part1 = np.sum(np.log(distance_x1_x2+1e-6), axis=-1)

            if self.only_xx:
                return part1.mean()
            else:
                part2 = self.numpy_phi_f(x1_samples).sum(axis=-1)\
                        + self.numpy_phi_f(x2_samples).sum(axis=-1)
                return part1.mean() + self.expectation_yy - part2.mean()

    def compute_average_q_and_bonus(self, obs, use_target_value=False):
        #batch_size = state.shape[0]
        new_obs = obs.expand(self.sample_number,*obs.shape)
        x, x_info = self.policy.action(
            new_obs, reparameterize=True, return_log_prob=True,
            )
        q = self.qf.value(new_obs.unsqueeze(-3), x.unsqueeze(-3), return_info=False, use_target_value=use_target_value)
        average_q = q.mean(0)

        if self.bonus_type == 'entropy':
            log_prob = x_info['log_prob']
            bonus = - log_prob.reshape(self.sample_number, -1, 1).mean(0)
            return average_q, bonus

        if self.bonus_type == 'variance':
            bonus = x.var(0).sum(-1, keepdim=True)
            return average_q, bonus

        elif 'phi' in self.bonus_type:
            x1 = x.reshape(1,self.sample_number,-1,self.action_size)
            x2 = x.reshape(self.sample_number,1,-1,self.action_size)
            distance_x1_x2 = (x1 - x2)**2
            if self.bonus_type == 'phi_power':
                part1 = ((distance_x1_x2+1e-6) ** self.exponent).sum(dim=-1, keepdim=True)
            elif self.bonus_type == 'phi_log':
                part1 = torch.log(distance_x1_x2+1e-6).sum(dim=-1, keepdim=True)
            part1 = (part1 * self.weight_matrix).sum(dim=[-3,-4])

            if self.only_xx:
                bonus = part1
            else:
                part2 = self.torch_phi_f(x).sum(dim=-1, keepdim=True)
                part2 = part2.reshape(self.sample_number, -1, 1).mean(0)
                bonus = part1 + self.expectation_yy - 2*part2

            return average_q, bonus
            


    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Alpha
        """
        average_q_new_actions, bonus = self.compute_average_q_and_bonus(obs)

        if self.use_automatic_bonus_tuning:
            alpha_loss = (self.log_alpha * (bonus - self.target_bonus).detach()).mean()
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

        average_q_next_actions, next_bonus = self.compute_average_q_and_bonus(next_obs,True)
        target_v_next_action = average_q_next_actions + alpha * next_bonus
        #target_v_next_action = average_q_next_actions
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_next_action
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

        policy_loss = -(average_q_new_actions + alpha * bonus).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Compute some statistics for eval
        """

        diagnostics = OrderedDict()
        diagnostics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
        diagnostics['Policy Q Loss'] = 0-np.mean(ptu.get_numpy(average_q_new_actions))
        diagnostics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        diagnostics['Averaged Bonus'] = np.mean(ptu.get_numpy(bonus))
        if self.use_automatic_bonus_tuning:
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
            self.eval_statistics.update(diagnostics)
        self._n_train_steps_total += 1
        
        return diagnostics

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        if self.use_automatic_bonus_tuning:
            ratio = min( (epoch+1)/self.end_reducing_exploration, 1 )
            self.target_bonus = self.start_target_bonus + ratio * (self.end_target_bonus - self.start_target_bonus)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf
        )

