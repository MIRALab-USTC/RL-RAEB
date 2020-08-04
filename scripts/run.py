import sys 
sys.path.insert(0, '/home/rl_shared/zhihaiwang/research/mbrl_sparse_reward')

from mbrl.utils.launch_utils import parse_cmd, run_experiments

if __name__ == '__main__':
    run_experiments(*parse_cmd())