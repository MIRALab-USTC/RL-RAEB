import abc
import gtimer as gt

from mbrl.algorithms.base_algorithm import RLAlgorithm
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.process import Progress, Silent
from mbrl.utils.misc_untils import format_for_process

from mbrl.utils.logger import logger
from mbrl.utils.normalizer import TransitionNormalizer



class BatchRLAlgorithm(RLAlgorithm):
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
            record_video_freq=50,
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


class ModelBasedBatchRLAlgorithm(RLAlgorithm):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            num_train_models_per_epoch,
            train_model_freq,
            model_normalize=False,
            max_path_length=1000,
            min_num_steps_before_training=0,
            silent = False,
            record_video_freq=50,
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

        self.train_model_freq = train_model_freq
        self.model_normalize = model_normalize
        self.num_train_models_per_epoch = num_train_models_per_epoch
        
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
                if i % self.train_model_freq == 0:
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


class VirtualLossBatchRLAlgorithm(RLAlgorithm):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            num_train_models_per_epoch,
            train_model_freq,
            max_path_length=1000,
            min_num_steps_before_training=0,
            silent = False,
            record_video_freq=50,
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
        self.train_model_freq = train_model_freq

        self.num_train_models_per_epoch = num_train_models_per_epoch
        
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
                self.virtual_pool.update_hash_table(samples)

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

    def _train_epoch(self, epoch):
        progress = self.process_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        
        # to do check 没有normalizer 是否ok？
        #normalizer = self.get_normalizer()
        #self.trainer.model.setup_normalizer(normalizer)
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
                    train_model_data = self.pool.sample_ensemble_batch(self.batch_size, self.trainer.model.ensemble_size)
                    params_model = self.trainer.train_model(train_model_data)
                    for k,v in params_model.items():
                        params[k] = v
                    # clear virtual hash table
                    #params['hash_table'] = self.virtual_pool.get_diagnostics()
                    if self.virtual_pool.clear:
                        self.virtual_pool.clear()
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



