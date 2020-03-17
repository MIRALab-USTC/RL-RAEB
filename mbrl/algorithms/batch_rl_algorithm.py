import abc
import gtimer as gt

if __name__ == "__main__":
    import sys
    import os
    mbrl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

from mbrl.algorithms.base_algorithm import RLAlgorithm
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.process import Progress, Silent
from mbrl.utils.misc_untils import format_for_process

class BatchRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            max_path_length=1000,
            min_num_steps_before_training=0,
            silent = False,
            item_dict_config={},
    ):
        super().__init__(num_epochs, item_dict_config)
        self.batch_size = batch_size
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.max_path_length = max_path_length

        self.process_class = Silent if silent else Progress
        self.collected_samples = 0
    
    def _sample(self, num_steps):
        if num_steps > 0:
            if hasattr(self.expl_collector, 'collect_new_paths'):
                paths = self.expl_collector.collect_new_paths(num_steps, self.max_path_length, False)
                self.pool.add_paths(paths)
            elif hasattr(self.expl_collector, 'collect_new_steps'):
                samples = self.expl_collector.collect_new_steps(num_steps, self.max_path_length, False)
                self.pool.add_samples(samples)

    def _before_train(self):
        self.start_epoch(-1)
        if hasattr(self, 'init_expl_policy'):
            with self.expl_collector.with_policy(self.init_expl_policy):
                self._sample(self.min_num_steps_before_training)
        else:
            self._sample(self.min_num_steps_before_training)
        self.end_epoch(-1)

    def _train_epoch(self, epoch):
        progress = self.process_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for _ in range(self.num_train_loops_per_epoch):
            self._sample(self.num_expl_steps_per_train_loop)
            gt.stamp('exploration sampling', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                train_data = self.pool.random_batch(self.batch_size)
                params = self.trainer.train(train_data)
                progress.set_description(format_for_process(params))
            gt.stamp('training', unique=False)
            self.training_mode(False)

        self.eval_collector.collect_new_paths(
            self.num_eval_steps_per_epoch,
            self.max_path_length,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')
        progress.close()

if __name__ == "__main__":
    import json
    from collections import OrderedDict
    from mbrl.utils.launch_utils import setup_logger, set_gpu_mode
    import mbrl.torch_modules.utils as ptu
    config_path = os.path.join(mbrl_dir, 'mbrl', 'algorithms', 'default_configs', 'sac.json')
    config = json.load(open(config_path, 'r'), object_pairs_hook=OrderedDict)
    setup_logger('test_data', base_log_dir='/home/qizhou/', variant=config)
    config.pop('launch_kwargs')
    kwargs = config.pop('algorithm')['kwargs']
    kwargs['item_dict_config'] = config
    set_gpu_mode(True)  # optionally set the GPU (default=False)
    algo = BatchRLAlgorithm(**kwargs)
    algo.to(ptu.device)
    algo.train()

