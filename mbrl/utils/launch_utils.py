import datetime
import argparse
import dateutil.tz
import pickle
import json
import os
import os.path as osp
from os.path import join
import random
import numpy as np
import torch
from collections import OrderedDict
import time

import mbrl
from mbrl.utils.logger import logger
import mbrl.torch_modules.utils as ptu


_mbrl_project_dir = join(os.path.dirname(mbrl.__file__), os.pardir)
_LOCAL_LOG_DIR = join(_mbrl_project_dir, 'data')

def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False

def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}

    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d

def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)

def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=0,
        base_log_dir=None,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log directories be under this directory.
    :param exp_id: The number of the specific experiment run within this experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id, seed)

    if base_log_dir is None:
        base_log_dir = _LOCAL_LOG_DIR

    log_dir = join(base_log_dir, exp_prefix, exp_name)

    if osp.exists(log_dir):
        logger.log("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to

        base_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """

    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = join(log_dir, tabular_log_file)
    text_log_path = join(log_dir, text_log_file)
    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)

    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir

def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time())%4096
    np.random.seed(seed)    
    random.seed(seed)    
    torch.manual_seed(seed) #cpu    
    torch.cuda.manual_seed_all(seed)  #并行gpu    
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致    
    torch.backends.cudnn.benchmark = True 
    return seed

def parse_cmd():
    p = argparse.ArgumentParser()
    p.add_argument('config_file', type=str)
    p.add_argument('--env_name', type=str)

    """
    add by ourself
    """
    p.add_argument('--intrinsic_coeff', type=float)
    p.add_argument('--max_step', type=int)
    p.add_argument('--int_coeff_decay', action='store_true')
    p.add_argument('--intrinsic_normal', action='store_true')
    p.add_argument('--min_num_steps_before_training', type=float, default=5000)

    p.add_argument('--layers_num', type=int)
    p.add_argument('--hidden_size', nargs='+', default=[128,128])

    
    p.add_argument('--base_log_dir', type=str)
    p.add_argument('--repeat', type=int)
    p.add_argument('--model_normalize', action='store_true')

    # alg args 
    p.add_argument('--max_path_length', type=int)
    p.add_argument('--num_eval_steps_per_epoch', type=int, default=8000)

    # simple hash pool
    p.add_argument('--hash_k', type=int, default=16)

    args, extras = p.parse_known_args()

    def foo(astr):
        if astr.startswith('--'):
            astr = astr[2:]
        elif astr.startswith('-'):
            astr = astr[1:]
        else:
            raise RuntimeError('Keys must start with \"--\" or \"-\".')

        return astr

    cmd_config = [[foo(k),v] for k,v in zip(extras[::2],extras[1::2])]

    if args.base_log_dir is not None:
        cmd_config.insert(0, ['experiment.base_log_dir', args.base_log_dir])

    if args.repeat is not None:
        cmd_config.insert(0, ['experiment.repeat', args.repeat])

    if args.env_name is not None:
        cmd_config.insert(0, ['type-environment.env_name', args.env_name])
    
    if args.intrinsic_coeff is not None:
        cmd_config.insert(0, ['class-Surprise_Based_SAC_Trainer.intrinsic_coeff', args.intrinsic_coeff])
    if args.max_step is not None:
        cmd_config.insert(0, ['class-Surprise_Based_SAC_Trainer.max_step', args.max_step])

    if args.intrinsic_normal is not None:
        cmd_config.insert(0, ['class-Surprise_Based_SAC_Trainer.intrinsic_normal', args.intrinsic_normal])
        cmd_config.insert(0, ['class-Vision_Surprise_SAC_Trainer.intrinsic_normal', args.intrinsic_normal])
    
    if args.max_path_length is not None:
        cmd_config.insert(0, ['class-batch_RL_algorithm.max_path_length', args.max_path_length])
    if args.num_eval_steps_per_epoch is not None:
        cmd_config.insert(0, ['class-batch_RL_algorithm.num_eval_steps_per_epoch', args.num_eval_steps_per_epoch])
    
    if args.hash_k is not None:
        cmd_config.insert(0, ['class-simple_pool_with_hash_state_action.hash_k', args.hash_k])

    if args.min_num_steps_before_training is not None:
        cmd_config.insert(0, ['class-batch_RL_algorithm.min_num_steps_before_training', args.min_num_steps_before_training])

    if args.layers_num is not None:
        cmd_config.insert(0, ['class-model_no_reward.layers_num', args.layers_num])
        cmd_config.insert(0, ['class-model_no_reward.hidden_size', args.hidden_size])
    
    if args.model_normalize is not None:
        cmd_config.insert(0, ['class-ModelBasedBatchRLAlgorithm.model_normalize', args.model_normalize])


    cmd_config = OrderedDict(cmd_config)
    return args.config_file, cmd_config

def try_eval(v):
    try:
        v = eval(v)
    except:
        pass
    return v

def _set_config_by_k_v(config, k, v):
    from mbrl.algorithms.utils import _visit_all_items
    v = try_eval(v)
    keys = k.split('.')
    if len(keys) == 2:
        if keys[0] == 'experiment':
            config['experiment'][keys[1]] = v
        elif keys[0][:5] == "type-":
            for _,item_type,_,kwargs in _visit_all_items(config):
                if item_type == keys[0][5:]:
                    kwargs[keys[1]] = v
        elif keys[0][:6] == "class-":
            for _,_,class_name,kwargs in _visit_all_items(config):
                if class_name == keys[0][6:]:
                    kwargs[keys[1]] = v
        else:
            for name,_,_,kwargs in _visit_all_items(config):
                if name == keys[0]:
                    kwargs[keys[1]] = v
    elif len(keys) == 1:
        config['experiment'][keys[0]] = v
    else:
        raise NotImplementedError
    return config


def update_config(config, cmd_config):
    for k,v in cmd_config.items():
        _set_config_by_k_v(config, k, v)

def get_config_from_file(config_path):
    config = json.load(open(config_path, 'r'))

    if 'base_config_file' in config:
        base_config_file = config.pop('base_config_file')
        base_config = get_config_from_file(base_config_file)
        base_config.update(config)
        config = base_config

    if 'cmd_config' in config:
        cmd_config = config.pop('cmd_config')
        update_config(config, cmd_config)

    return config

def run_experiments(config_path, cmd_config):
    if osp.isdir(config_path):
        for file_name in os.listdir(config_path):
            if file_name[-5:] == '.json':
                json_path = osp.join(config_path, file_name)
                _run_experiments(json_path, cmd_config)
    else:
        _run_experiments(config_path, cmd_config)
        
def _run_experiments(config_path, cmd_config):
        config = get_config_from_file(config_path)
        update_config(config, cmd_config)
        if 'exp_prefix' not in config['experiment']:
            exp_prefix = osp.basename(config_path)
            exp_prefix = exp_prefix.split('.')[0]
            config['experiment']['exp_prefix'] = exp_prefix
        repeat = config['experiment'].pop('repeat', 1)
        
        for _ in range(repeat):
            run_single_experiment(config)

def run_single_experiment(config):
    import copy
    from mbrl.algorithms.utils import get_item
    config = copy.deepcopy(config)
    experiment_kwargs = config['experiment']
    seed = experiment_kwargs.get('seed', None)
    seed = set_global_seed(seed)
    experiment_kwargs['seed'] = seed
    use_gpu = experiment_kwargs.get('use_gpu', False)
    ptu.set_gpu_mode(use_gpu)
    print(ptu.device)

    logger.reset()
    variant = copy.deepcopy(config)
    experiment_kwargs.pop('use_gpu', None)
    experiment_kwargs.pop('tag', None)
    actual_log_dir = setup_logger(
        variant=variant,
        **experiment_kwargs
    )

    config.pop('experiment')
    algo = config.pop('algorithm')
    algo_class = algo['class']
    algo_kwargs = algo['kwargs']
    algo_kwargs['item_dict_config'] = config

    algo = get_item('algorithm', algo_class, algo_kwargs)
    algo.to(ptu.device)
    algo.train()
