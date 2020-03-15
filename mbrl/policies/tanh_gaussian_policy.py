
if __name__ == "__main__":
    import sys
    import os
    mbrl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

from mbrl.policies.base_policy import RandomPolicy
from mbrl.torch_modules.policies import MeanLogstdGaussianPolicyModule, TanhPolicyModule
from mbrl.utils.logger import logger
from torch import nn

class TanhGaussianPolicy(nn.Module, RandomPolicy):
    def __init__( self, 
                  env, 
                  hidden_layers=[300,300], 
                  activation='relu', 
                  obs_processor=None,
                  deterministic=False,
                  policy_name='tanh_gaussian_policy',):
        nn.Module.__init__(self)
        RandomPolicy.__init__(self, env, obs_processor, deterministic)
        assert len(self.processed_obs_shape) == 1 and len(self.action_shape) == 1
        gaussian = MeanLogstdGaussianPolicyModule( self.processed_obs_shape[0], 
                                                        self.action_shape[0],
                                                        hidden_layers, 
                                                        activation, 
                                                        policy_name)
        self.module = TanhPolicyModule(gaussian)
        
    def _action( self, 
                 obs, 
                 return_info=True,
                 return_log_prob=False,
                 **kwargs
                ):
            if return_info:
                action, info = self.module(obs, return_info=True, return_log_prob=return_log_prob, **kwargs)
                if return_log_prob:
                    info['log_prob'] = info['log_prob'].sum(dim=-1, keepdim=True)
                return action, info
            else:
                return self.module(obs, return_info=False, **kwargs)

    def _log_prob(self, obs, action):
        return self.module.log_prob(obs, action).sum(dim=-1, keepdim=True)

    def save(self, save_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir)
    
    def load(self, load_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.load(save_dir)

    def get_snapshot(self):
        return self.module.get_snapshot()


if __name__ == '__main__':
    import gym
    import torch
    from torch import optim
    from mbrl.environments.utils import make_vector_env
    from mbrl.collectors.path_collector import SimplePathCollector
    from mbrl.pools.simple_pool import SimplePool
    from mbrl.processors.normalizer import Normalizer
    import mbrl.torch_modules.utils as ptu
    env = make_vector_env('MountainCarContinuous-v0', n_env=2, max_length=20)
    normalizer = Normalizer(env.observation_shape)
    pi = TanhGaussianPolicy(env, obs_processor=normalizer, hidden_layers=[32,32])
    collector = SimplePathCollector(env, pi)
    pool = SimplePool(env)
    mean_std = pool.dataset_mean_std['observations'] 
    pi_optimizer = optim.Adam(
        pi.parameters(),
        lr=1e-1,
    )
    for i in range(300):
        print("iteration", i)
        for j in range(2):
            paths = collector.collect_new_paths(128, discard_incomplete_paths=False)
            pool.add_paths(paths)
            normalizer.set_mean_std_np(mean_std.mean, mean_std.std)
            print(normalizer.mean_std_np()[1])
            print("pool size:", pool._size, end='\t')
            batch = pool.random_batch(32)
            x = batch["observations"]
            print("x shape:", x.shape, end='\t')
            x = ptu.FloatTensor(x)
            actions, info = pi.action(x,reparameterize=True)
            if i % 10 == 0 and j == 0:
                print(actions)
                print(info)
            loss = torch.mean(actions**2)
            print(ptu.get_numpy(loss))
            pi_optimizer.zero_grad()
            loss.backward()
            pi_optimizer.step()

    
