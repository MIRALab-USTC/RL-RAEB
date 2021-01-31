import numpy as np
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from ipdb import set_trace

# from normalizer import TransitionNormalizer

# code from Model-Based Active Exploration https://github.com/nnaisense/max
def swish(x):
    return x * torch.sigmoid(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnsembleDenseLayer(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, non_linearity='leaky_relu'):
        """
        linear + activation Layer
        there are `ensemble_size` layers
        computation is done using batch matrix multiplication
        hence forward pass through all models in the ensemble can be done in one call

        weights initialized with xavier normal for leaky relu and linear, xavier uniform for swish
        biases are always initialized to zeros

        Args:
            n_in: size of input vector
            n_out: size of output vector
            ensemble_size: number of models in the ensemble
            non_linearity: 'linear', 'swish' or 'leaky_relu'
        """

        super().__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        # 权重模块
        # different initialize method
        for weight in weights:
            if non_linearity == 'swish':
                nn.init.xavier_uniform_(weight)
            elif non_linearity == 'leaky_relu':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'tanh':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'linear':
                nn.init.xavier_normal_(weight)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        # different activation function
        if non_linearity == 'swish':
            self.non_linearity = swish
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif non_linearity == 'linear':
            self.non_linearity = lambda x: x

    def forward(self, inp):
        # torch.baddbmm 矩阵乘法
        #op = torch.baddbmm(self.biases, inp, self.weights)
        # set_trace()
        op = torch.matmul(inp, self.weights) + self.biases
        return self.non_linearity(op)


class Model(object, metaclass=abc.ABCMeta):
    def __init__(self, env, hidden_size, layers_num, ensemble_size, non_linearity):
        self.env = env
        self.dim_action = env.action_space.shape[0]
        self.dim_state = env.observation_space.shape[0]

        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.ensemble_size = ensemble_size
        self.non_linearity = non_linearity
        self.device = device

        # default normalizer is None
        self.normalizer = None

    def setup_noramlizer(self, normalizer):
        # 外部传入一个normalizer, 据此设置model的normalizer
        pass

    def _preprocess_inputs(self, states):
        # 归一化处理输入状态
        pass
    def _preprocess_targets(self, target_states):
        # 归一化处理label 状态
        pass
    def _post_process_outputs(self, delta_mean, var):
        # 将输出反处理到真实状态
        pass

    def forward(self, normalize_input_obs, normalize_input_action, unnormalize_output_obs):
        # predict with raw datas
        pass
    def forward_all(self, states, actions):
        pass
    def loss(self, states, actions, state_deltas, training_noise_stdev=0):
        # loss function for training models
        pass
        
    def get_diagnostics():
        pass


class ModelNoReward(nn.Module, Model):
    def __init__(self, env, hidden_size, layers_num, ensemble_size, non_linearity):
        nn.Module.__init__(self)
        Model.__init__(self, env, hidden_size, layers_num, ensemble_size, non_linearity)
        self.dim_action = env.action_space.shape[0]
        self.dim_state = env.observation_space.shape[0]
        #assert layers_num >= 1 
        layers = []
        for lyr_idx in range(layers_num + 1):
            if lyr_idx == 0:
                lyr = EnsembleDenseLayer(self.dim_action + self.dim_state , self.hidden_size[0], self.ensemble_size, non_linearity=self.non_linearity)
                layers.append(lyr)
            if 0 < lyr_idx < self.layers_num:
                lyr = EnsembleDenseLayer(self.hidden_size[lyr_idx - 1], self.hidden_size[lyr_idx], self.ensemble_size, non_linearity=self.non_linearity)
                layers.append(lyr)
            if lyr_idx == self.layers_num:
                lyr = EnsembleDenseLayer(self.hidden_size[lyr_idx], self.dim_state + self.dim_state, self.ensemble_size, non_linearity='linear')
                layers.append(lyr)

        self.layers = nn.Sequential(*layers)
        self.min_log_var = -5
        self.max_log_var = -1
    def setup_normalizer(self, normalizer):
        # 外部传入一个normalizer, 据此设置model的normalizer
        self.normalizer = normalizer
        #print(f"normalizer_state_mean: {self.normalizer.state_mean}")

    def _preprocess_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is None:
            return states, actions
        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)

        return states, actions

    def _preprocess_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is None:
            return state_deltas
        
        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_outputs(self, delta_mean, var):
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            var = self.normalizer.denormalize_state_delta_vars(var)
        return delta_mean, var

    def forward(self, states, actions):
        # predict with raw datas
        normalized_states, normalized_actions = self._preprocess_inputs(states, actions)
        #set_trace()
        normalized_delta_mean, normalized_var = self._propagate_network(normalized_states, normalized_actions)
        #set_trace()


        delta_mean, var = self._post_process_outputs(normalized_delta_mean, normalized_var)
        # 差量预测
        next_state_mean = delta_mean + states.to(self.device)
        return next_state_mean, var

    def forward_all(self, states, actions):
        """
        predict next state mean and variance of a batch of states and actions for all models.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch tensor): (batch size, dim_state)
            actions (torch tensor): (batch size, dim_action)

        Returns:
            next state means (torch tensor): (batch size, ensemble_size, dim_state)
            next state variances (torch tensor): (batch size, ensemble_size, dim_state)
        """
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_state_means, next_state_vars = self(states, actions)
        return next_state_means.transpose(0, 1), next_state_vars.transpose(0, 1)
        
    def _propagate_network(self, normalized_states, normalized_actions):
        inp = torch.cat((normalized_states, normalized_actions), dim=2)
        op = self.layers(inp)

        delta_mean, log_var = torch.split(op, op.size(2) // 2, dim=2)
        #print(f"model_delta_mean: {delta_mean}")
        #print(f"log_var: {log_var}")

        log_var = torch.sigmoid(log_var)      # in [0, 1]
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        var = torch.exp(log_var)              # normal scale, not log

        return delta_mean, var

    def loss(self, states, actions, state_deltas, training_noise_stdev=0):
        # loss function for training models
        """
        compute loss given states, actions and state_deltas

        the loss is actually computed between predicted state delta and actual state delta, both in normalized space

        Args:
            states (torch tensor): (ensemble_size, batch size, dim_state)
            actions (torch tensor): (ensemble_size, batch size, dim_action)
            state_deltas (torch tensor): (ensemble_size, batch size, dim_state)
            training_noise_stdev (float): noise to add to normalized state, action inputs and state delta outputs

        Returns:
            loss (torch 0-dim tensor): `.backward()` can be called on it to compute gradients
        """

        states, actions = self._preprocess_inputs(states, actions)
        targets = self._preprocess_targets(state_deltas)

        if not np.allclose(training_noise_stdev, 0):
            states += torch.randn_like(states) * training_noise_stdev
            actions += torch.randn_like(actions) * training_noise_stdev
            targets += torch.randn_like(targets) * training_noise_stdev

        #set_trace()
        mu, var = self._propagate_network(states, actions)      # delta and variance
        #set_trace()
        # negative log likelihood
        loss = (mu - targets) ** 2 / (var + 1e-6) + torch.log(var+1e-6)
        #set_trace()

        loss = torch.mean(loss)
        return loss
        
    def get_diagnostics():
        pass

    def sample(self, mean, var):
        """
        sample next state, given next state mean and variance

        Args:
            mean (torch tensor): any shape
            var (torch tensor): any shape

        Returns:
            next state (torch tensor): same shape as inputs
        """

        return Normal(mean, torch.sqrt(var)).sample()

