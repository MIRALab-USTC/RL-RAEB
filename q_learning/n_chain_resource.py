import gym
from gym import spaces
from gym.utils import seeding

class NChainResource(gym.Env):
    """n-Chain environment
        code based on openai gym n-Chain environment
    """
    def __init__(self, n=5, small=2, large=10):
        self.n = 2 * n # 2n states 
        #self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(3) # action 0 right  ||  action 1 left  ||  action 2 release resources 
        self.observation_space = spaces.Discrete(self.n) # 0 - n/2 -1 with resources  || n/2 - n-1 without resources 
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #if self.np_random.rand() < self.slip:
        #    action = not action  # agent slipped, reverse action taken
        reward = 0.0
        if action == 0:  # go right no reward
            if self.state != (self.n/2 - 1) and self.state != (self.n - 1):
                self.state += 1
        
        elif action == 1:   # go left no reward
            if self.state != 0 and self.state != self.n/2:
                self.state -= 1
        else:
            if self.state < self.n/2:  # state with resources
                if self.state == (self.n/2 - 1):
                    reward = 100
                self.state += (self.n/2)
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass


class TabularResource(gym.Env):
    """Tabular environment
        code based on openai gym n-Chain environment
    """
    def __init__(self, n=5, small=2, large=10):
        self.n = 2 * n # 2n states 
        #self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(3) # action 0 right  ||  action 1 left  ||  action 2 release resources 
        self.observation_space = spaces.Discrete(self.n) # 0 - n/2 -1 with resources  || n/2 - n-1 without resources 
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #if self.np_random.rand() < self.slip:
        #    action = not action  # agent slipped, reverse action taken
        reward = 0.0
        if action == 0:  # go right no reward
            if self.state != (self.n/2 - 1) and self.state != (self.n - 1):
                self.state += 1
        
        elif action == 1:   # go left no reward
            if self.state != 0 and self.state != self.n/2:
                self.state -= 1
        else:
            if self.state < self.n/2:  # state with resources
                if self.state == (self.n/2 - 1):
                    reward = 100
                self.state += int(self.n/2)
        done = False
        return int(self.state), reward, done, {}

    def reset(self):
        self.state = 0
        return int(self.state)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass