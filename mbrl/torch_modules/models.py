import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import mbrl.torch_modules.utils as ptu
from mbrl.utils.misc_utils import to_list

class EnsembleModelModule(MLP):
    def __init__(
            self, 
            env, 
            hidden_layers, 
            predict_reward=False,
            deterministic=True, 
            nonlinearity='swish', 
            ensemble_size=5, 
            model_name='probabilistic_ensemble_model'
            ):
        assert ensemble_size is not None and ensemble_size > 0
        self.predict_reward = predict_reward
        self.deterministic = deterministic

        if predict_reward:
            self.prediction_size = env.observation_space.shape + 1
        else:
            self.prediction_size = env.observation_space.shape

        if deterministic:
            super(PEModule, self).__init__(
                env.observation_space.shape,
                self.prediction_size,
                hidden_layers,
                nonlinearity,
                ensemble_size,
                model_name,
            )
        else:
            super(PEModule, self).__init__(
                env.observation_space.shape,
                self.prediction_size * 2,
                hidden_layers,
                nonlinearity,
                ensemble_size,
                model_name,
            )
            self.max_log_std = nn.Parameter(ptu.ones(1, self.prediction_size) * 0.5)
            self.min_log_std = nn.Parameter(-ptu.ones(1, self.prediction_size) * 10.0)

    def forward(
            self,
            x,
            deterministic=True,
            elite_indices=None,

            reparameterize=True,
            return_log_prob=False,
            return_mean_std=False,
            return_ensemble=False,
    ):
        """
        NOTE: log_prob is dimension-wise

        :param x: Observation
        :param sample_number: if None, return a matrix with shape (None, prediction_size)
                              if (int), return a tensor with shape (sample_number, None, prediction_size)
        :param deterministic: 
        :param elite_indices: indicate which models to predict the next obs
        :param return_ensemble: If True, return an extra ensemble of prediction
        :return: 
        """
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i != len(self.fcs) - 1:
                x = self.act_f(x)
        
        if self.deterministic:
            predictions = x
        else:
            means, log_stds = torch.chunk(x,2,-1)
            if deterministic:
                predictions = means
            else:
                log_stds = self.max_log_std - F.softplus(self.max_log_std - log_stds)
                log_stds = self.min_log_std + F.softplus(log_stds - self.min_log_std)
                stds = torch.exp(log_stds)
                normal = Normal(means, stds)
                if reparameterize:
                    predictions = (
                        means +
                        stds *
                        Normal(
                            ptu.zeros_like(means),
                            ptu.ones_like(means)
                        ).sample()
                    )
                    predictions.requires_grad_()
                else:
                    predictions = normal.sample()
                log_probs = normal.log_prob(predictions)

        results = [predictions]

        if (not self.deterministic) and (not deterministic):
            if self.return_log_prob:
                results.append(log_probs)
            if self.return_mean_std:
                results += [means, stds]

        input_dim = len(x.shape)

        if input_dim == 2 and (not return_ensemble):
            if elite_indices is None:
                elite_indices = np.arange(self.ensemble_size)
            index = int(np.random.choice(elite_indices))
            results = [item[index] for item in results]

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)