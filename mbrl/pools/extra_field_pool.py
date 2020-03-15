import numpy as np
import warnings

if __name__ == "__main__":
    import sys
    import os
    mbrl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

from mbrl.pools.simple_pool import SimplePool
class ExtraFieldPool(SimplePool):
    def __init__(self, env, max_size=1e6, extra_fields={}, compute_mean_std=True):
        super(ExtraFieldPool, self).__init__(env, max_size, compute_mean_std)
        self.extra_fields = {}
        self.required_samples = {}
        self.extra_fields_stop = {}
        self.extra_fields_size = {}
        self.add_extra_fields(extra_fields)
    
    def add_extra_fields(self, extra_fields):
        self.extra_fields.update(extra_fields)
        for k,v in extra_fields.items():
            assert k not in self.fields
            if k in self.extra_fields:
                warnings.warn('Add a same extra_field [%s]. It will cover the old one.'%k)
            self.required_samples[k] = 0
            self.extra_fields_stop[k] = 0
            self.extra_fields_size[k] = 0
            self.initialize_field(k, v) 

    def add_samples(self, samples):
        new_sample_size = super(ExtraFieldPool, self).add_samples(samples)
        for key in self.extra_fields:
            self.required_samples[key] += new_sample_size
    
    def _check_keys(self, keys):
        if keys is None:
            keys = list(self.fields.keys()) + list(self.extra_fields.keys())
        for k in keys:
            if k in self.extra_fields:
                if self.required_samples[k] > 0:
                    raise RuntimeError("the %s data is not aligned with the sampled data."%(k))
            else:
                assert k in self.fields
        return keys
        
    def _update_single_extra_field(self, key, value):
        assert key in self.extra_fields
        if self.compute_mean_std:
            self.dataset_mean_std[key].update(value)
        new_sample_size = len(value)
        max_size = self.max_size
        assert new_sample_size <= self.required_samples[key]
        self.required_samples[key] -= new_sample_size
        stop = self.extra_fields_stop[key]
        self.extra_fields_stop[key] = new_stop = (stop + new_sample_size) % max_size
        cur_size = self.extra_fields_size[key]
        self.extra_fields_size[key] = min(max_size, cur_size + new_sample_size)
        if stop + new_sample_size >= max_size:
            self.dataset[key][stop:max_size] = value[:max_size-stop]
            self.dataset[key][:new_stop] = value[new_sample_size-new_stop:]
        else:
            self.dataset[key][stop:new_stop] = value

    def update_extra_fields(self, data):
        for k,v in data.items():
            assert k in self.extra_fields
            self._update_single_extra_field(k, v)

            
if __name__ == "__main__":
    from mbrl.environments.utils import make_vector_env
    from mbrl.policies.base_policy import UniformlyRandomPolicy
    from mbrl.collectors.step_collector import SimpleStepCollector
    from mbrl.collectors.path_collector import SimplePathCollector
    env = make_vector_env('HalfCheetah-v2', n_env=2, max_length=3)
    policy = UniformlyRandomPolicy(env)
    collector = SimpleStepCollector(env, policy)
    extra_filed = {'rewards_copy': {'shape':(1,), 'type':np.float}}
    replay_pool = ExtraFieldPool(env,12,extra_filed)
    collector.start_epoch()
    samples = collector.collect_new_steps(6,6,True)
    replay_pool.add_samples(samples)
    data = replay_pool.get_unprocessed_data('copy',keys=['rewards'])
    data['rewards_copy'] = data.pop('rewards')
    replay_pool.update_extra_fields(data)
    replay_pool.update_process_flag('copy',len(data['rewards_copy']))
    print(replay_pool.get_data(keys=['rewards','rewards_copy']))
    print(replay_pool.dataset['rewards_copy'])
    print('\n\n')
    samples = collector.collect_new_steps(6,6,True)
    replay_pool.add_samples(samples)
    data = replay_pool.get_unprocessed_data('copy',keys=['rewards'])
    data['rewards_copy'] = data.pop('rewards')
    replay_pool.update_extra_fields(data)
    replay_pool.update_process_flag('copy',len(data['rewards_copy']))
    print(replay_pool.get_data(keys=['rewards','rewards_copy']))
    print(replay_pool.dataset['rewards_copy'])
    print('\n\n')
    samples = collector.collect_new_steps(6,6,True)
    replay_pool.add_samples(samples)
    data = replay_pool.get_unprocessed_data('copy',keys=['rewards'])
    data['rewards_copy'] = data.pop('rewards')
    replay_pool.update_extra_fields(data)
    replay_pool.update_process_flag('copy',len(data['rewards_copy']))
    print(replay_pool.get_data(keys=['rewards','rewards_copy']))
    print(replay_pool.dataset['rewards_copy'])
    print('\n\n')
    print(replay_pool.random_batch(2))
    samples = collector.collect_new_steps(6,6,True)
    replay_pool.add_samples(samples)
    print(replay_pool.get_data(keys=['rewards','rewards_copy']))
    print(replay_pool.dataset['rewards_copy'])
    print('\n\n')
    collector.end_epoch()
