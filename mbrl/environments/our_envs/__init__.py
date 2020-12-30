
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
    id='ContinuousMountaincarResource-v0',
    entry_point='mbrl.environments.our_envs.mountain_car_resources:ResourceMountainCarEnv',
    max_episode_steps=1000
)
register(
    id='AntMaze-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnv',
    max_episode_steps=200
)
register(
    id='AntMazeDenseReward-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnvDenseReward',
    max_episode_steps=1000
)

register(
    id='AntMazeEnvForwardReward-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnvForwardReward',
    max_episode_steps=500
)

register(
    id='AntMazeEnvGoal-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnvGoal',
    kwargs={
        'goal_pos': {'x': 4,'y': 0}
    },
    max_episode_steps=500,
)

register(
    id='AntMazeEnvGoalForwardReward-v0',
    entry_point='mbrl.environments.our_envs.ant:AntMazeEnvGoalForwardReward',
    kwargs={
        'goal_pos': {'x': 4,'y': 0}
    },
    max_episode_steps=500,
)

## ant maze resource 
register(
    id='AntMazeResource-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_env:MazeEnv',
    max_episode_steps=500
)

# cargo 4 beta 5 block 3
register(
    id='AntMazeResourceBlock3-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [3]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock3-v1',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 10,
        'reward_block': [3]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock3-v2',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 50,
        'reward_block': [3]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock3-v3',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 100,
        'reward_block': [3]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock4-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock2Cargo1-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 1,
        'beta': 5,
        'reward_block': [2]
    },
    max_episode_steps=500,
)

register(
    id='AntMazeResourceBlock24-v0',
    entry_point='mbrl.environments.our_envs.ant_maze_resource:AntMazeResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [2,3,4]
    },
    max_episode_steps=500,
)


### ant corridor
register(
    id='AntCorridorEnv-v3',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [3,4]
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorEnv-v4',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [4,5]
    },
    max_episode_steps=500,
)

### ant cooridor resources

register(
    id='AntCorridorResourceEnv-v3',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [6,7],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnv-v4',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [5,6],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnv-v5',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnv-v52',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 10
    },
    max_episode_steps=500,
)


register(
    id='AntCorridorResourceEnv-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [7,8],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnv-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [8,9],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnv-v2',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [9,10],
        'reward': 100
    },
    max_episode_steps=500,
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
    "ant_maze_dense_reward_v0": "AntMazeDenseReward-v0",
    'ant_maze_forward_reward': "AntMazeEnvForwardReward-v0",
    'ant_maze_goal':"AntMazeEnvGoal-v0",
    'ant_maze_goal_forward_reward': "AntMazeEnvGoalForwardReward-v0",
    'ant_maze_resource_block3': "AntMazeResourceBlock3-v0",
    'ant_maze_resource_block3_beta10': "AntMazeResourceBlock3-v1",
    'ant_maze_resource_block3_beta50': "AntMazeResourceBlock3-v2",
    'ant_maze_resource_block3_beta100': "AntMazeResourceBlock3-v3",
    'ant_maze_resource_block4': "AntMazeResourceBlock4-v0",
    'ant_maze_resource_block2_cargo1': "AntMazeResourceBlock2Cargo1-v0",
    'ant_maze_resource_block2_4': "AntMazeResourceBlock24-v0",
    'ant_corridor_resource_env_goal_7_v0': 'AntCorridorResourceEnv-v0',
    'ant_corridor_resource_env_goal_8_v0': 'AntCorridorResourceEnv-v1',
    'ant_corridor_resource_env_goal_9_v0': 'AntCorridorResourceEnv-v2',
    'ant_corridor_resource_env_goal_6_v0': 'AntCorridorResourceEnv-v3',
    'ant_corridor_resource_env_goal_5_v0': 'AntCorridorResourceEnv-v4',
    'ant_corridor_resource_env_goal_4_v0': 'AntCorridorResourceEnv-v5',
    'ant_corridor_resource_env_goal_4_v1': 'AntCorridorResourceEnv-v52',
    'ant_corridor_env_3': "AntCorridorEnv-v3",
    'ant_corridor_env_4': "AntCorridorEnv-v4"
}
