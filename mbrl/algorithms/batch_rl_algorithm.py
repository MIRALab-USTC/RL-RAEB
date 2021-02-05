import abc
import gtimer as gt
import torch
import os

from mbrl.algorithms.base_algorithm import RLAlgorithm
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.process import Progress, Silent
from mbrl.utils.misc_untils import format_for_process

from mbrl.utils.logger import logger
from mbrl.utils.normalizer import TransitionNormalizer
from ipdb import set_trace
class BatchRLAlgorithm(RLAlgorithm):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            save_pool=True,
            max_path_length=1000,
            min_num_steps_before_training=0,
            silent = False,
            record_video_freq=50,
            save_model_freq=2,
            item_dict_config={},
    ):
        super().__init__(num_epochs, item_dict_config)
        self._need_snapshot.append('trainer')
        self.batch_size = batch_size
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.max_path_length = max_path_length
        self.record_video_freq = record_video_freq
        self.save_model_freq = save_model_freq
        self.save_pool = save_pool
        self.log_models_dir = os.path.join(logger._snapshot_dir, "models")
        if not os.path.exists(self.log_models_dir):
            os.mkdir(self.log_models_dir)

        self.process_class = Silent if silent else Progress
        self.collected_samples = 0
    
    def _sample(self, num_steps):
        if num_steps > 0:
            if hasattr(self.expl_collector, 'collect_new_paths'):
                paths = self.expl_collector.collect_new_paths(num_steps, self.max_path_length, True)
                self.pool.add_paths(paths)
            elif hasattr(self.expl_collector, 'collect_new_steps'):
                samples = self.expl_collector.collect_new_steps(num_steps, self.max_path_length, True)
                self.pool.add_samples(samples)

    def _before_train(self):
        self.start_epoch(-1)
        if hasattr(self, 'init_expl_policy'):
            with self.expl_collector.with_policy(self.init_expl_policy):
                self._sample(self.min_num_steps_before_training)
        else:
            self._sample(self.min_num_steps_before_training)
        self.end_epoch(-1)
    
    def _end_epoch(self, epoch):
        from mbrl.collectors.utils import rollout
        if epoch % self.record_video_freq == 0 and hasattr(self, 'video_env'):
            self.video_env.set_video_name("epoch{}".format(epoch))
            logger.log("rollout to save video...")
            rollout(self.video_env, self.eval_policy, max_path_length=self.max_path_length, use_tqdm=True)
        gt.stamp('save video', unique=False)

        if epoch % self.save_model_freq == 0:
            save_filename_policy = os.path.join(self.log_models_dir, f'epoch_{epoch}_policy.pkl')
            torch.save(self.eval_policy, save_filename_policy)
            save_filename_qf = os.path.join(self.log_models_dir, f'epoch_{epoch}_q.pkl')
            torch.save(self.trainer.qf, save_filename_qf)

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

    def _after_train(self):
        if self.save_pool:
            dataset = self.pool.dataset
            logger.save_replay_pool(dataset)

class RNDRLAlgorithm(BatchRLAlgorithm):
    def __init__(
        self,
        train_model_freq,
        num_train_models_per_epoch,
        **kwargs
    ):
        self.train_model_freq = train_model_freq
        self.num_train_models_per_epoch = num_train_models_per_epoch
        BatchRLAlgorithm.__init__(self, **kwargs)

    def _train_epoch(self, epoch):
        progress = self.process_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        
        for i in range(max(self.num_train_loops_per_epoch, self.num_train_models_per_epoch)):
            # sample a transition
            self._sample(self.num_expl_steps_per_train_loop)
            gt.stamp('exploration sampling', unique=False)
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                train_data = self.pool.random_batch(self.batch_size)
                params = self.trainer.train(train_data)
                # 每 train_model_freq train 一步
                if i % self.train_model_freq == 0:
                    # train random network
                    # train_data: dict s,a,n_s,r,d
                    params_model = self.trainer.train_model(train_data)
                    for k,v in params_model.items():
                        params[k] = v

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
    
class ModelBasedBatchRLAlgorithm(BatchRLAlgorithm):
    def __init__(
            self,
            train_model_freq,
            num_train_models_per_epoch,
            model_normalize,
            **kwargs
        ):
        BatchRLAlgorithm.__init__(self, **kwargs)

        self.train_model_freq = train_model_freq
        self.model_normalize = model_normalize
        self.num_train_models_per_epoch = num_train_models_per_epoch

    def _train_epoch(self, epoch):
        progress = self.process_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        
        # to do check 没有normalizer 是否ok？
        if self.model_normalize:
            normalizer = self.get_normalizer()
            self.trainer.model.setup_normalizer(normalizer)
        for i in range(max(self.num_train_loops_per_epoch, self.num_train_models_per_epoch)):
            # sample a transition
            self._sample(self.num_expl_steps_per_train_loop)
            gt.stamp('exploration sampling', unique=False)
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                train_data = self.pool.random_batch(self.batch_size)
                params = self.trainer.train(train_data)
                # 每 train_model_freq train 一步
                if (i+1) % self.train_model_freq == 0:
                    # set_trace()
                    train_model_data = self.pool.sample_ensemble_batch(self.batch_size, self.trainer.model.ensemble_size)
                    params_model = self.trainer.train_model(train_model_data)
                    for k,v in params_model.items():
                        params[k] = v
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

    def get_normalizer(self):
        mean_std_dict = self.pool.get_mean_std()
        normalizer = TransitionNormalizer(mean_std_dict)
        return normalizer

