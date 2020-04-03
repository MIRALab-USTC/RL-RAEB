from gym.envs.registration import register

register(
    id='MBRLHalfCheetah-v0',
    entry_point='mbrl.environments.gym_envs.half_cheetah:HalfCheetahEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLWalker2d-v0',
    entry_point='mbrl.environments.gym_envs.walker2d:Walker2dEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLSwimmer-v0',
    entry_point='mbrl.environments.gym_envs.swimmer:SwimmerEnv',
    kwargs={'frame_skip': 4},
    max_episode_steps=1000,
)

register(
    id='MBRLAnt-v0',
    entry_point='mbrl.environments.gym_envs.ant:AntEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

register(
    id='MBRLHopper-v0',
    entry_point='mbrl.environments.gym_envs.hopper:HopperEnv',
    kwargs={'frame_skip': 5},
    max_episode_steps=1000,
)

env_name_to_gym_registry_dict = {
    "mbrl_half_cheetah": "MBRLHalfCheetah-v0",
    "mbrl_swimmer": "MBRLSwimmer-v0",
    "mbrl_ant": "MBRLAnt-v0",
    "mbrl_hopper": "MBRLHopper-v0",
    "mbrl_walker2d": "MBRLWalker2d-v0",
    "half_cheetah": "HalfCheetah-v2",
    "swimmer": "Swimmer-v2",
    "ant": "Ant-v2",
    "hopper": "Hopper-v2",
    "walker2d": "Walker2d-v2",
    "humanoid": "Humanoid-v2",
}
