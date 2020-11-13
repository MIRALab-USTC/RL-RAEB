import gym
from gym import spaces
from gym.utils import seeding

from n_chain_resource import NChainResource



class NChainMultiResources(NChainResource):

    def __init__(self, n=5, resources_num=5):
        self.n = (resources_num + 1) * n # 2n states
        self.single_resource_n = n
        self.resources_num = resources_num
        #self.slip = slip  # probability of 'slipping' an action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(3) # action 0 right  ||  action 1 left  ||  action 2 release resources 
        self.observation_space = spaces.Discrete(self.n) # 0 - single_resource_n -1 with resources  || n/2 - n-1 without resources 
        self.seed()
        self.get_reward = True


    def step(self, action):
        assert self.action_space.contains(action)
        #if self.np_random.rand() < self.slip:
        #    action = not action  # agent slipped, reverse action taken
        # 假设每一层都在最后一步释放资源才能获得奖励
        reward = 0.0
        if action == 0:  # go right no reward
            if (self.state + 1) % self.single_resource_n != 0:
                self.state += 1
        
        elif action == 1:   # go left no reward
            if self.state % self.single_resource_n != 0:
                self.state -= 1
        else:
            if self.get_reward:  # state with resources
                if self.state == (self.n-self.single_resource_n -1):
                    reward = 100
                elif (self.state + 1) % self.single_resource_n != 0:
                    self.get_reward = False
            if self.n - self.state > self.single_resource_n:    
                self.state += self.single_resource_n
        done = False
        return self.state, reward, done, {}


    def reset(self):
        self.state = 0
        self.get_reward = True
        return self.state