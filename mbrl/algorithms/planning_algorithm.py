import abc
import gtimer as gt
import numpy as np
import copy 
import torch

from mbrl.algorithms.base_algorithm import RLAlgorithm
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.process import Progress, Silent
from mbrl.utils.misc_untils import format_for_process

from mbrl.utils.logger import logger
from mbrl.utils.normalizer import TransitionNormalizer
from mbrl.utils.utilities import get_utility_measure

from mbrl.environments.image_envs.imagination import Imagination
from mbrl.models.base_model_without_reward import ModelNoReward
from mbrl.trainers.sac import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlanningImagineAlgorithm(RLAlgorithm):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            num_train_models_per_epoch,
            num_steps_per_epoch=25,
            num_train_model_epochs=50,
            train_model_batch_size=256,
            train_model_freq=25,

            imagine_policy_horizon=50,
            imagine_policy_actors=128,
            imagine_measure_mode='renyi_div',
            policy_episodes=50,
            policy_warm_up_episodes=3,
            n_exploration_steps=25000,

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

        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_train_model_epochs = num_train_model_epochs
        self.train_model_batch_size = train_model_batch_size

        self.imagine_policy_horizon = imagine_policy_horizon
        self.imagine_policy_actors = imagine_policy_actors
        self.imagine_measure_mode = imagine_measure_mode
        self.measure = get_utility_measure(imagine_measure_mode, 0, 0.1)
        self.policy_episodes = policy_episodes
        self.policy_warm_up_episodes = policy_warm_up_episodes

        self.n_exploration_steps = n_exploration_steps

        self.env_current_state = None

    def _before_train(self):
        # 256 step warm up replay pool
        self.start_epoch(-1)
        self.end_epoch(-1)

    def _sample(self, num_steps):
        if num_steps > 0:
            if hasattr(self.expl_collector, 'collect_new_paths'):
                paths = self.expl_collector.collect_new_paths(num_steps, self.max_path_length, True)
                self.pool.add_paths(paths)
            elif hasattr(self.expl_collector, 'collect_new_steps'):
                samples = self.expl_collector.collect_new_steps(num_steps, self.max_path_length, True)
                self.pool.add_samples(samples)
                self.env_current_state = samples['next_observations'][-1]


    def _train_model(self):
        for _ in range(self.num_train_model_epochs):
            tr_loss = self._train_model_epoch()
        return tr_loss

    def _train_model_epoch(self):
        losses = []
        # 从pool 中过一批 分别训练 ensemble 个model
        for batch in self.pool.shuffer_and_random_batch_model(self.train_model_batch_size, self.trainer.model.ensemble_size):
            params_model = self.trainer.train_model(batch)
            losses.append(params_model['Model Loss'])
        return np.mean(losses)

    def _train_epoch(self, epoch):
        # to do:
        # train policy in imagine MDP
        # use another pool 
        # train model
        progress = self.process_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)

        env = self.expl_collector._env
        state = env.reset()
        
        # train policy in imagine MDP        
        mdp = Imagination(horizon=self.imagine_policy_horizon, n_actors=self.imagine_policy_actors, model=self.trainer.model, measure=self.measure)
        agent = None
        print(f"train_model_device: {self.trainer.model.device}")
        for step_num in range(1, self.n_exploration_steps + 1):
            # to do check 没有normalizer 是否ok？
            print(f"train_model_device: {self.trainer.model.device}")
            if agent is not None:
                print(f"agent: {agent.device}")
            gt.stamp('start_training', unique=False)
            if step_num > self.min_num_steps_before_training:
                # planning to action
                normalizer = self.get_normalizer()
                self.trainer.model.setup_normalizer(normalizer)
                if mdp is None:
                    mdp = Imagination(horizon=self.imagine_policy_horizon, n_actors=self.imagine_policy_actors, model=self.trainer.model, measure=self.measure)
                mdp.update_init_state(state)
                if agent is not None:
                    # 从当前state 选择动作
                    action, mdp, agent, policy_value = self.get_action(mdp, agent)
                else:
                    # 如果agent 是None,需要重新planning 训练 策略网络

                    agent = self.reset_agent()
                    ep_returns = []

                    for i in range(self.policy_episodes):
                        #gt.stamp('policy training', unique=False)
                        progress.update()
                        if i < self.policy_warm_up_episodes:
                            # 随机收集一个episode数据 到 想象pool中
                            ep_return = agent.episode(env=mdp, warm_up=True)
                        else:
                            ep_return = agent.episode(env=mdp, warm_up=False)
                        ep_returns.append(ep_return)
                    # 复制 虚拟数据训练的网络的参数 到 真实policy中
                    #self.policy.load_state_dict(self.trainer.agent.policy.state_dict())
                    #self.policy = copy.deepcopy(agent.policy)
            else:
                action = env.action_space.sample()
            
            # true env
            next_state, reward, done, info = env.step(action)
            # buffer 没有添加 reward
            sample = {
                'observations': state,
                'actions': action,
                'next_observations': next_state,
                'rewards': reward,
                'terminals': done,
                'env_infos': info,
                'agent_infos': None,
            }
            self.pool.add_samples(sample)

            if done:
                print(f"step: {step_num}\tepisode complete")
                agent = None
                mdp = None
                next_state = env.reset()

            state = next_state

            if step_num < self.min_num_steps_before_training:
                continue

            episode_done = done
            train_at_end_of_episode = (self.train_model_freq is np.inf)
            time_to_update = ((step_num % self.train_model_freq) == 0)
            just_finished_warm_up = (step_num == self.min_num_steps_before_training)
            if (train_at_end_of_episode and episode_done) or time_to_update or just_finished_warm_up:
                #self.reset_model()
                # train model from real data 
                tr_loss = self._train_model()
                # discard old solution and MDP as models changed
                params = {
                    "model loss": tr_loss,
                }
                progress.set_description(format_for_process(params))
                mdp = None
                agent = None

            if step_num % self.record_video_freq == 0:
                self.eval_collector.collect_new_paths(
                    self.num_eval_steps_per_epoch,
                    self.max_path_length,
                    discard_incomplete_paths=True,
                )
                # 每一个 step 是一个 epoch
                self.end_epoch(step_num)


    
    def _end_epoch(self, epoch):
        # record video 
        from mbrl.collectors.utils import rollout
        if epoch % self.record_video_freq == 0 and hasattr(self, 'video_env'):
            self.video_env.set_video_name("epoch{}".format(epoch))
            logger.log("rollout to save video...")
            rollout(self.video_env, self.eval_policy, max_path_length=self.max_path_length, use_tqdm=True)
        gt.stamp('save video', unique=False)

    def get_normalizer(self):
        mean_std_dict = self.pool.get_mean_std()
        normalizer = TransitionNormalizer(mean_std_dict)
        return normalizer

    def reset_model(self):
        env, hidden_size, layers_num, ensemble_size, non_linearity = self.trainer.model.env, self.trainer.model.hidden_size, self.trainer.model.layers_num, self.trainer.model.ensemble_size, self.trainer.model.non_linearity
        model = ModelNoReward(env, hidden_size, layers_num, ensemble_size, non_linearity)
        model = model.to(device)
        self.trainer.model = model
        normalizer = self.get_normalizer()
        self.trainer.model.setup_normalizer(normalizer)

    def get_action(self, mdp, agent):
        current_state = mdp.reset()
        actions = agent(current_state, eval=True)
        action = actions[0].detach().data.cpu().numpy()
        policy_value = torch.mean(agent.get_state_value(current_state)).item()
        return action, mdp, agent, policy_value
    
    def reset_agent(self):        
        d_state, d_action, replay_size, batch_size, n_updates, n_hidden, gamma, alpha, lr, tau = self.trainer.agent.d_state, self.trainer.agent.d_action, self.trainer.agent.replay_size, self.trainer.agent.batch_size, self.trainer.agent.n_updates, self.trainer.agent.n_hidden, self.trainer.agent.gamma, self.trainer.agent.alpha, self.trainer.agent.lr, self.trainer.agent.tau
        agent = SAC(d_state, d_action, replay_size, batch_size, n_updates, n_hidden, gamma, alpha, lr, tau)
        agent = agent.to(device)
        #agent.setup_normalizer(self.trainer.model.normalizer)
        return agent