import gym
from gym import spaces
from gym.utils import seeding

class NChainOursEnv(gym.Env):
    """n-Chain environment
        code based on openai gym n-Chain environment
    """
    def __init__(self, n=5, small=2, large=10):
        self.n = n
        #self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 1  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #if self.np_random.rand() < self.slip:
        #    action = not action  # agent slipped, reverse action taken
        if action:  # go right
            if self.state + 1 >= (self.n-1):
                reward = self.large
            else:
                reward = 0
            self.state = self.state + 1 if self.state < (self.n-1) else self.state

        else: 
            # go left
            if self.state <= 1:  
                reward = self.small
                
            else:
                reward = 0
            self.state = self.state - 1 if self.state > 0 else self.state
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 1
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass