from mbrl.environments.gym import env_name_to_gym_registry_dict
import gym

def set_env_seed(env, seed=None):
    while True:
        if hasattr(env, "seed"):
            env_seed = env.seed(seed)
            if env_seed != None and env_seed != []:
                print("Set the seed of current environment successfully.\nSeed: %d"%env_seed[0])
                return env_seed[0]
        if hasattr(env, "wrapped_env"): 
            env = env.wrapped_env
        elif hasattr(env, "_wrapped_env"): 
            env = env._wrapped_env
        elif hasattr(env, "env"): 
            env = env.env
        else:
            break
    print('WARNING: Fail to set the seed of current environment.')
    return -1

def env_name_to_gym_registry(env_name):
    if env_name in env_name_to_gym_registry_dict:
        return env_name_to_gym_registry_dict[env_name]
    return env_name

def make_gym_env(env_name):
    return gym.make(env_name_to_gym_registry(env_name)).env

def make_vector_env(env_name, **kwargs):
    from mbrl.environments.normalized_vector_env import NormalizedVectorEnv
    return NormalizedVectorEnv(env_name, **kwargs)
