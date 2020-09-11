import gym
import random 
import argparse

import numpy as np 
from n_chain_resource import NChainResource

class base_q_learning():
    def __init__(
        self,
        env,
        num_steps_per_episode=200,
        gamma=0.99,
        alpha=0.1
    ):
        self.env = env
        self.num_steps_per_episode = num_steps_per_episode
        self.state_num = self.env.n
        print(f"action_space: {self.env.action_space.n}")
        self.action_num = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        # init q
        self.q = np.zeros((self.state_num, self.action_num))
        
        self.cur_state = 0
        
        self.avg_returns = []
    
    def before_train(self):
        pass

    def train(self, epochs):
        pass

    def train(self, epochs):
        for epoch in range(epochs):
            # reset env
            self.cur_state = self.env.reset()
            for i in range(self.num_steps_per_episode):
                a = self.select_action()
                next_s, reward, done, info = self.env.step(a)
                if done:
                    break
                self.update_q_table(self.cur_state, a, next_s, reward)
                self.cur_state = next_s
            avg_return = self.evaluate()
            self.avg_returns.append(avg_return)
            print(f"epoch: {epoch}, avg_return: {avg_return}")

    def evaluate(self):
        q = self.q
        env = self.env
        s = env.reset()
        avg_return = 0.0 
        for i in range(self.num_steps_per_episode):
            a = np.argmax(q[s]) 
            next_s, reward, done, info = env.step(a)
            avg_return += reward
            if done: 
                break
            s = int(next_s)
        return avg_return

    def update_q_table(self, s, a, next_s, r):
        # update q value
        s = int(s)
        a = int(a)
        next_s = int(next_s)
        q_target = r + self.gamma * np.max(self.q[int(next_s)])


        self.q[s][a] = self.q[s][a] + self.alpha * (q_target - self.q[s][a])

    def select_action(self):
        pass

    def end_train(self):
        pass

    def argmax(self, s):
        s = int(s)
        if np.count_nonzero(self.q[s]) == 0:
            action = random.randrange(self.action_num)
        
        else:
            action = np.argmax(self.q[s])
        return action
    

class epsilon_greedy_q_learning(base_q_learning):
    def __init__(
        self,
        env,
        num_steps_per_episode=100,
        gamma=0.99,
        alpha=0.1,
        epsilon=0.1
    ):
        base_q_learning.__init__(self, env=env, num_steps_per_episode=num_steps_per_episode, gamma=gamma, alpha=alpha)

        self.epsilon = epsilon

 
    def select_action(self):
        ran_num = random.random()
        if ran_num <= self.epsilon:
            # epsilon randomly choose action
            action = random.randrange(self.action_num)
            return int(action)
        
        else:
            # greedily choose action
            # print(f"type_state: {type(self.cur_state)}")
            action = self.argmax(self.cur_state)
            return int(action)

class UCB_q_learning(base_q_learning):
    def __init__(
        self,
        env,
        num_steps_per_episode=100,
        gamma=0.99,
        alpha=0.1
    ):
        base_q_learning.__init__(self, env, num_steps_per_episode, gamma, alpha)
        self.count_table = np.ones((self.state_num, self.action_num))

    def select_action(self):
        s = int(self.cur_state)
        # 1 / 向量 OK 吗？
        a = int(np.argmax(self.q[s] + 1/self.count_table[s]))
        self.count_table[s][a] += 1
        return int(a) 


class WUCB_q_learning(base_q_learning):
    def __init__(
        self,
        env,
        num_steps_per_episode=100,
        gamma=0.99,
        alpha=0.1,
        k=100
    ):
        base_q_learning.__init__(self, env, num_steps_per_episode, gamma, alpha)
        self.count_table = np.ones((self.state_num, self.action_num))

        self.f = np.zeros((self.state_num, self.action_num))
        for i in range(int(self.state_num/2)):
            self.f[i][2] = 1

        self.k = k 

    def I(self, s):
        if s < self.state_num/2:
            return 1
        else:
            return 0

    def select_action(self):
        s = self.cur_state
        s = int(s) 
        a = np.argmax(self.q[s] + 1/self.count_table[s])
        a = int(a)
        self.count_table[s][a] += (self.k * self.f[s][a] + 1)
        return int(a) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q_learning')
    parser.add_argument('--alg', type=str, default='epsilon')
    parser.add_argument('--env_name', type=str, default='chain_resource')
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    if args.env_name == 'chain_resource':
        env = NChainResource(n=20)
        print(f"env_n: {env.n}")
    
    #agent = None
    if args.alg == 'epsilon':
        agent = epsilon_greedy_q_learning(env)
        print(f"start_training")
        agent.train(args.epochs)
    elif args.alg == 'UCB':
        agent = UCB_q_learning(env)
        print(f"start_training")
        agent.train(args.epochs)
    elif args.alg == 'WUCB':
        agent = WUCB_q_learning(env)
        print(f"start_training")
        agent.train(args.epochs)

    
    

    

