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

if __name__ == "__main__":
    import sys
    mbrl_dir = osp.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(mbrl_dir)

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
        seed = random.randint(0,65535)
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
    args, extras = p.parse_known_args()

    def foo(astr):
        if astr.startswith('--'):
            astr = astr[2:]
        elif astr.startswith('-'):
            astr = astr[1:]
        else:
            raise RuntimeError('Keys must start with \"--\" or \"-\".')

        astr = astr.replace('-','_')
        return astr

    cmd_config = {foo(k):v for k,v in zip(extras[::2],extras[1::2])}
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
        for name,_,_,kwargs in _visit_all_items(config):
            if name == keys[0]:
                kwargs[keys[1]] = v
    elif len(keys) == 1:
        config['experiment'][keys[0]] = v
    else:
        raise NotImplementedError
    return config


def update_config_by_cmd(config, cmd_config):
    for k,v in cmd_config.items():
        _set_config_by_k_v(config, k, v)

def run_experiment(config_path, cmd_config):
    import copy
    from mbrl.algorithms.utils import get_item

    config = json.load(open(config_path, 'r'))
    update_config_by_cmd(config, cmd_config)

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

if __name__ == '__main__':
    run_experiment(*parse_cmd())
    

