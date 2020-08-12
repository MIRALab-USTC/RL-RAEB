import numpy as np

import torch

from gym.spaces import Box

import warnings


class Imagination:
    def __init__(self, model, n_actors, horizon, measure):
        """
        Imaginary MDP

        Args:
            model: models.Model object
            n_actors: number of parallel episodes
            horizon: length of the episode
            measure: the reward function
        """

        self.model = model
        self.n_actors = n_actors
        self.horizon = horizon
        self.measure = measure
        self.ensemble_size = model.ensemble_size

        self.action_space = Box(low=-1.0, high=1.0, shape=(n_actors, self.model.dim_action), dtype=np.float32)
        self.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))

        self.init_state = None
        self.states = None
        self.steps = None

    def step(self, actions):
        n_act = self.n_actors
        es = self.ensemble_size

        actions = actions.to(self.model.device)

        # get next state distribution for all models
        with torch.no_grad():
            # input (n_act, ensemble, dim_state) 
            # output (n_act, ensemble, dim_state)
            next_state_means, next_state_vars = self.model.forward_all(self.states, actions)    # shape: (n_actors, ensemble_size, d_state)

        # (0,1,....,n_actors)
        i = torch.arange(n_act).to(self.model.device)
        # 从ensemble_size随机抽取n_actor个数字
        j = torch.randint(es, size=(n_act,)).to(self.model.device)

        next_states = self.model.sample(next_state_means[i, j], next_state_vars[i, j])          # shape: (n_actors, d_state)

        if torch.any(torch.isnan(next_states)).item():
            warnings.warn("NaN in sampled next states!")

        if torch.any(torch.isinf(next_states)).item():
            warnings.warn("Inf in sampled next states!")

        # compute measure
        measures = self.measure(self.states,                                         # shape: (n_actors, d_state)
                                actions,                                             # shape: (n_actors, d_action)
                                next_states,                                         # shape: (n_actors, d_state)
                                next_state_means,                                    # shape: (n_actors, ensemble_size, d_state)
                                next_state_vars,                                     # shape: (n_actors, ensemble_size, d_state)
                                self.model)

        self.states = next_states
        self.steps += 1
        done = False
        if self.steps >= self.horizon:
            done = True

        return next_states, measures, done, {}

    def reset(self):
        states = torch.from_numpy(self.init_state).float()
        #states = states.unsqueeze(0)
        states = states.repeat(self.n_actors, 1)
        states = states.to(self.model.device)
        self.steps = 0
        self.states = states                    # shape: (n_actors, d_state)
        return states

    def update_init_state(self, state):
        self.init_state = state
