
from gym.envs.registration import register
register(
    id='MultiGoal2DRandomReset-v0',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':True,
        'goals':[[0,5], [0,-5], [5,0], [-5,0]]
    },
    max_episode_steps=30,
)
register(
    id='MultiGoal2D-v0',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':False,
        'goals':[[0,5], [0,-5], [5,0], [-5,0]]
    },
    max_episode_steps=30,
)

register(
    id='MultiGoal2DRandomReset-v1',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':True,
        'goals':[[4,4], [-4,-4], [4,-4], [-4,4]]
    },
    max_episode_steps=30,
)
register(
    id='MultiGoal2D-v1',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':False,
        'goals':[[4,4], [-4,-4], [4,-4], [-4,4]]
    },
    max_episode_steps=30,
)

register(
    id='MultiGoal2DRandomReset-v2',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':True,
        'goals':[[5,0], [-5,0]],
        'bound': [8,5]
    },
    max_episode_steps=30,
)
register(
    id='MultiGoal2D-v2',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':False,
        'goals':[[5,0], [-5,0]],
        'bound': [8,5]
    },
    max_episode_steps=30,
)

register(
    id='NChainOur-v0',
    entry_point='mbrl.environments.our_envs.n_chain:NChainOursEnv',
    kwargs={
        'n':50,
        'small':1,
        'large': 100
    },
    max_episode_steps=59,
)

register(
    id='MagellanAnt-v2',
    entry_point='mbrl.environments.our_envs.ant:MagellanAntEnv',
    max_episode_steps=300
)


register(
    id='MagellanHalfCheetah-v2',
    entry_point='mbrl.environments.our_envs.half_cheetah:MagellanHalfCheetahEnv',
    max_episode_steps=100
)

register(
    id='MagellanSparseMountainCar-v0',
    entry_point='mbrl.environments.our_envs.mountain_car:MagellanSparseContinuousMountainCarEnv',
    max_episode_steps=500
)

register(
    id='AntMazeResource-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_env:MazeEnv',
    max_episode_steps=500
)
register(
    id='ContinuousMountaincarResource-v0',
    entry_point='mbrl.environments.our_envs.mountain_car_resources:ResourceMountainCarEnv',
    max_episode_steps=1000
)
register(
    id='AntMaze-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnv',
    max_episode_steps=1000
)
register(
    id='AntMazeDenseReward-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnvDenseReward',
    max_episode_steps=1000
)

env_name_to_gym_registry_dict = {
    "mbrl_half_cheetah": "MBRLHalfCheetah-v0",
    "mbrl_cheetah": "MBRLHalfCheetah-v0",
    "mbrl_swimmer": "MBRLSwimmer-v0",
    "mbrl_ant": "MBRLAnt-v0",
    "mbrl_hopper": "MBRLHopper-v0",
    "mbrl_walker2d": "MBRLWalker2d-v0",
    "half_cheetah": "HalfCheetah-v2",
    "cheetah": "HalfCheetah-v2",
    "swimmer": "Swimmer-v2",
    "ant": "Ant-v2",
    "hopper": "Hopper-v2",
    "walker2d": "Walker2d-v2",
    "humanoid": "Humanoid-v2",
    "mountaincar": "MountainCarContinuous-v0",
    "ant_maze": "MagellanAnt-v2",
    "mountaincar_sparse": "MagellanSparseMountainCar-v0",
    "cheetah_sparse": "MagellanHalfCheetah-v2",
    "n_chain":"NChainOur-v0",
    "ant_maze_resource": "AntMazeResource-v0",
    "continuous_car_resource": "ContinuousMountaincarResource-v0",
    "ant_maze_v0": "AntMaze-v0",
    "ant_maze_dense_reward_v0": "AntMazeDenseReward-v0"
}
