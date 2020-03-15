import numpy as np
import warnings
from collections import OrderedDict

if __name__ == "__main__":
    import sys
    import os
    mbrl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

from mbrl.utils.mean_std import RunningMeanStd
from mbrl.pools.base_pool import Pool
from mbrl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mbrl.collectors.utils import path_to_samples
class SimplePool(Pool):
    def __init__(self, env, max_size=1e6, compute_mean_std=True):
        self._env = env
        self.compute_mean_std = compute_mean_std
        self.max_size = int(max_size)
        o_shape = self._observation_shape = self._env.observation_space.shape
        a_shape = self._action_shape = self._env.action_space.shape
        self.fields = {
            'observations': {
                'shape': o_shape,
                'type': np.float,
            },
            'next_observations': {
                'shape': o_shape,
                'type': np.float,
            },
            'actions': {
                'shape': a_shape,
                'type': np.float,
            },
            'rewards': {
                'shape': (1,),
                'type': np.float,
            },
            'terminals': {
                'shape': (1,),
                'type': np.float,
            },
        }
        self.dataset = {}
        if self.compute_mean_std:
            self.dataset_mean_std = {}
        for k, v in self.fields.items():
            self.initialize_field(k, v) 
        self._size = 0
        self._stop = 0
        self.unprocessed_stop={}
        self.unprocessed_size={}

    def get_size(self):
        return self._size
    
    def initialize_field(self, field_name, field_info):
        self.dataset[field_name] = np.empty((int(self.max_size), *field_info['shape']), dtype=field_info['type'])
        if self.compute_mean_std:
            self.dataset_mean_std[field_name] = RunningMeanStd(field_info['shape'])

    def random_batch(self, batch_size, keys=None):
        keys = self._check_keys(keys)
        return _random_batch_independently(self.dataset, batch_size, self._size, keys)

    def shuffer_and_random_batch(self, batch_size, keys=None):
        keys = self._check_keys(keys)
        for batch in _shuffer_and_random_batch(self.dataset, batch_size, self._size, keys):
            yield batch

    def _update_single_field(self, key, value):
        assert key in self.fields
        if self.compute_mean_std:
            self.dataset_mean_std[key].update(value)
        new_sample_size = len(value)
        max_size = self.max_size
        stop = self._stop
        new_stop = (stop + new_sample_size) % max_size
        if stop + new_sample_size >= max_size:
            self.dataset[key][stop:max_size] = value[:max_size-stop]
            self.dataset[key][:new_stop] = value[new_sample_size-new_stop:]
        else:
            self.dataset[key][stop:new_stop] = value

    def add_paths(self, paths):
        self.add_samples(path_to_samples(paths))

    def get_data(self, keys=None):
        keys = self._check_keys(keys)
        data = {}
        for k in keys:
            data[k] = self.dataset[k][:self._size]
        return data
            
    def get_unprocessed_data(self, tag, keys=None):
        if keys is None:
            keys = list(self.fields.keys())
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
            self.unprocessed_stop[tag] = self._stop
        stop = self.unprocessed_stop[tag]
        size = self.unprocessed_size[tag]
        data = {}
        for key in keys:
            assert key in self.fields
            temp_data = self.dataset[key]
            if size > stop:
                data[key] = np.concatenate((temp_data[stop-size:], temp_data[:stop]))
            else:
                data[key] = temp_data[stop-size:stop]
        return data

    def update_process_flag(self, tag, process_num):
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
            self.unprocessed_stop[tag] = self._stop
        assert process_num <= self.unprocessed_size[tag]
        self.unprocessed_size[tag] -= process_num
    
    def add_samples(self, samples):
        for k in self.fields:
            v = samples[k]
            self._update_single_field(k,v)
        stop = self._stop
        new_sample_size = len(samples[k])
        max_size = self.max_size
        self._stop = new_stop = (stop + new_sample_size) % max_size
        self._size = min(max_size, self._size + new_sample_size)
        for tag in self.unprocessed_stop:
            self.unprocessed_stop[tag] = new_stop
            unprocessed_size = self.unprocessed_size[tag] + new_sample_size
            if unprocessed_size > max_size:
                warnings.warn("unprocessed_size > max_size")
                self.unprocessed_size[tag] = max_size
            else:
                self.unprocessed_size[tag] = unprocessed_size
        return new_sample_size
    
    def _check_keys(self, keys):
        if keys is None:
            keys = list(self.fields.keys())
        for k in keys:
            assert k in self.fields
        return keys
        
    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

        
if __name__ == "__main__":
    from mbrl.environments.utils import make_vector_env
    from mbrl.policies.base_policy import UniformlyRandomPolicy
    from mbrl.collectors.step_collector import SimpleStepCollector
    from mbrl.collectors.path_collector import SimplePathCollector
    env = make_vector_env('HalfCheetah-v2', n_env=3, max_length=5)
    policy = UniformlyRandomPolicy(env)
    # collector = SimpleStepCollector(env, policy)
    # replay_pool = SimplePool(env,10)
    # collector.start_epoch()
    # samples = collector.collect_new_steps(6,6,True)
    # replay_pool.add_samples(samples)
    # #print(replay_pool.get_data())
    # print(replay_pool.get_unprocessed_data('test',keys=['rewards']))
    # print(replay_pool.get_unprocessed_data('test2'))
    # replay_pool.update_process_flag('test',6)
    # replay_pool.update_process_flag('test2',3)
    # print('\n\n')
    # print(replay_pool.get_unprocessed_data('test',keys=['rewards']))
    # print(replay_pool.get_unprocessed_data('test2'))
    # samples=collector.collect_new_steps(6,5,True)
    # replay_pool.add_samples(samples)
    # print('\n\n')
    # print(replay_pool.get_unprocessed_data('test',keys=['rewards']))
    # print(replay_pool.get_unprocessed_data('test2'))
    # print(replay_pool.random_batch(3, keys=['rewards', 'actions']))
    # for batch in replay_pool.shuffer_and_random_batch(3, keys=['rewards']):
    #     print(batch)
    env = make_vector_env('HalfCheetah-v2', n_env=2, max_length=3)
    policy = UniformlyRandomPolicy(env)
    collector = SimplePathCollector(env, policy)
    replay_pool = SimplePool(env,12)
    paths = collector.collect_new_paths(6,6,True)
    replay_pool.add_paths(paths)
    print(replay_pool.get_data(keys=['terminals', 'rewards']))
    paths = collector.collect_new_paths(6,6,True)
    replay_pool.add_paths(paths)
    print(replay_pool.get_data(keys=['terminals', 'rewards']))
    paths = collector.collect_new_paths(6,6,True)
    replay_pool.add_paths(paths)
    print(replay_pool.get_data(keys=['terminals', 'rewards']))
    #print(replay_pool.get_data())
    # print('\n\n')
    # print(collector.collect_new_steps(3,5,True))
    # print('\n\n')
    # print(collector.collect_new_steps(3,5,True))
    # print('\n\n')
    # print(collector._epoch_paths)
    # print(collector.get_diagnostics())
    collector.end_epoch()