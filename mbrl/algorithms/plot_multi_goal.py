import abc
import gtimer as gt
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import os

from mbrl.utils.logger import logger
from mbrl.algorithms.batch_rl_algorithm import BatchRLAlgorithm
from mbrl.environments.our_envs.multi_goal import *

_POINTS = np.array([[0,0],
                   [2.5,2.5],
                   [-2.5,-2.5],
                   [2.5,-2.5],
                   [-2.5,2.5],
                   [2.5,0],
                   [-2.5,0],
                   [0,2.5],
                   [0,-2.5]])
NUMBER = 512
POINTS = np.repeat(_POINTS,NUMBER,axis=0).reshape(NUMBER,-1,2)

def get_s_a():
    n_state = _POINTS.shape[0]
    n_a = 258
    a_x = np.linspace(-1, 1, n_a)
    a_y = np.linspace(-1, 1, n_a)
    a_x, a_y = np.meshgrid(a_x, a_y)
    a = np.concatenate([a_x.reshape(a_x.shape+(1,)), a_y.reshape(a_y.shape+(1,))], axis=-1)
    a = np.tile(a,(n_state,1,1,1))
    s = np.repeat(_POINTS,n_a*n_a,axis=0).reshape(n_state, n_a, n_a, 2)
    return s, a

STATES, ACTIONS = get_s_a()
    


def draw(goals, bound):
    n = 256
    x = np.linspace(-bound[0], bound[0], n)
    y = np.linspace(-bound[1], bound[1], n)
    X,Y = np.meshgrid(x, y)

    # Basic contour plot
    # fig, ax = plt.subplots()
    CS = plt.contour(X, Y, f_reward(X,Y,goals), levels=[0.1,0.2,0.5,1,2,3,4,5,6,7,8,9], linewidths=1)

    # Recast levels to new class
    CS.levels = [ftos(val) for val in CS.levels]

    plt.scatter(goals[:,0], goals[:,1], c='r')
    plt.clabel(CS, CS.levels, inline=True, fontsize=6) # , CS.levels
    #plt.grid()

class PlotMultiGoal(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            **alg_kwarg
    ):
        super().__init__(**alg_kwarg)
    
    def _before_train(self):
        self.plot_path = os.path.join(logger._snapshot_dir, '2d-plot')
        os.makedirs(self.plot_path)
        self.start_epoch(-1)
        if hasattr(self, 'init_expl_policy'):
            with self.expl_collector.with_policy(self.init_expl_policy):
                self._sample(self.min_num_steps_before_training)
        else:
            self._sample(self.min_num_steps_before_training)
        self.end_epoch(-1)
        

    def _end_epoch(self, epoch):
        from mbrl.collectors.utils import rollout
        bound = self.eval_env.bound +0.5
        plt.figure(figsize=bound)
        if epoch % self.record_video_freq == 0:
            for i in range(12):
                path = rollout(self.eval_env, self.policy, max_path_length=self.max_path_length, use_tqdm=False)
                obs = path['observations']
                obs = obs.reshape((-1,2))
                x=obs[:,0]
                y=obs[:,1]
                plt.plot(x,y)
            plt.xlim(-bound[0],bound[0])
            plt.ylim(-bound[1],bound[1])
            draw(self.eval_env.goals,bound)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.savefig(os.path.join(self.plot_path, '%d-2d-multi-goal.png'%epoch))
            plt.close()

            actions, _ = self.policy.action_np(POINTS)
            q, _ = self.qf.value_np(STATES,ACTIONS)
            for i in range(_POINTS.shape[0]):
                plt.figure(figsize=(10,10))
                plt.contourf(ACTIONS[0,:,:,0],ACTIONS[0,:,:,1], q[i,:,:,0], 12, alpha=0.75)
                CS = plt.contour(ACTIONS[0,:,:,0],ACTIONS[0,:,:,1], q[i,:,:,0], 12, colors='black', linewidths=1)
                plt.rcParams['xtick.direction'] = 'in'
                plt.rcParams['ytick.direction'] = 'in'
                plt.scatter(actions[:,i,0],actions[:,i,1],color='red',alpha=0.6)     
                plt.xlim(-1,1)
                plt.ylim(-1,1)
                plt.savefig(os.path.join(self.plot_path, '%d-q-%d.png'%(epoch,i)))
                plt.close()

        
