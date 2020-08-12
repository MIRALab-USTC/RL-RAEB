from mbrl.environments.our_envs import env_name_to_gym_registry_dict

#from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
import gym
import warnings

def env_name_to_gym_registry(env_name):
    if env_name in env_name_to_gym_registry_dict:
        return env_name_to_gym_registry_dict[env_name]
    return env_name

def make_gym_env(env_name, seed):
    #if 'ant_maze' in env_name:
    #    env = AntMazeEnv()
    #else:
    env = gym.make(env_name_to_gym_registry(env_name)).env
    env.seed(seed)
    return env

def get_make_fn(env_name, seed):
    def make():
        #if 'ant_maze' in env_name:
        #    env = AntMazeEnv()
        #else:
        env = gym.make(env_name_to_gym_registry(env_name)).env
        env.seed(seed)
        return env
    return make

def get_make_fns(env_name, seeds, n_env=1):
    if seeds is None:
        seeds = [None] * n_env
    elif len(seeds) != n_env:
        warnings.warn('the length of the seeds is different from n_env')

    make_fns = [get_make_fn(env_name, seed) for seed in seeds]
    return make_fns

def get_state_block(state):
    x = state[2].item()
    y = state[3].item()

    if -1 < x < 1:
        x_block = 'low'
    elif 1 < x < 3:
        x_block = 'mid'
    elif 3 < x < 5:
        x_block = 'high'
    else:
        raise Exception

    if -1 < y < 1:
        y_block = 'left'
    elif 1 < y < 3:
        y_block = 'center'
    elif 3 < y < 5:
        y_block = 'right'
    else:
        raise Exception

    if x_block == 'low' and y_block == 'left':
        return 0
    elif x_block == 'low' and y_block == 'center':
        return 1
    elif x_block == 'low' and y_block == 'right':
        return 2
    elif x_block == 'mid' and y_block == 'right':
        return 3
    elif x_block == 'high' and y_block == 'right':
        return 4
    elif x_block == 'high' and y_block == 'center':
        return 5
    elif x_block == 'high' and y_block == 'left':
        return 6


def rate_buffer(buffer):
    visited_blocks = [get_state_block(state) for state in buffer.dataset['observations']]
    n_unique = len(set(visited_blocks))
    return n_unique