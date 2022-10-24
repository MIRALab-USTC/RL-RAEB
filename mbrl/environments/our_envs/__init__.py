
from gym.envs.registration import register

"""
Organization
    MultiGoal
    NChain
    Ant
    Mountain Car
    Racing Car
    Cheetah
    Swimmer
    Done
"""

# MultiGoal
######################################
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
######################################

# NChain (resource)
######################################

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
######################################

# Ant (resource)
######################################

register(
    id='MagellanAnt-v2',
    entry_point='mbrl.environments.our_envs.ant:MagellanAntEnv',
    max_episode_steps=300
)

## ant maze
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

register(
    id='AntCorridorEnv-v5',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [5,6]
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorEnv-v6',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [6,7]
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
    id='AntCorridorResourceEnvBeta1-v5',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 0.5
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2-v2',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 2.0
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2-v3',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 1.5
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2cargo2-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 2,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2cargo8-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 8,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorResourceEnvV2cargo16-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorResourceEnvV2',
    kwargs={
        'cargo_num': 16,
        'beta': 1,
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

register(
    id='RewardAntCorridorResourceEnv-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:RewardAntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'goal_reward': 10
    },
    max_episode_steps=500,
)

register(
    id='NoRewardAntCorridorResourceEnv-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:NoRewardAntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)
######################################

# Mountain Car (resource)
######################################
register(
    id='ResourceMountainCar-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=200,
    kwargs={
        'seed': None,
        'cargo_num': 10
    },
)

register(
    id='ResourceMountainCarV2-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnvV2',
    max_episode_steps=200,
    kwargs={
        'seed': None,
        'cargo_num': 10,
        "beta": 1
    },
)

register(
    id='NoRewardResourceMountainCar-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:NoRewardResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 10
    },
)

register(
    id='ResourceMountainCar-v8',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 15
    },
)

register(
    id='ResourceMountainCar-v4',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 5
    },
)

register(
    id='ResourceMountainCar-v5',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 2
    },
)

register(
    id='ResourceMountainCar-v6',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 200
    },
)

register(
    id='ResourceMountainCar-v7',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 100
    },
)

register(
    id='ResourceMountainCar-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 50
    },
)

register(
    id='ResourceMountainCar-v3',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 25
    },
)

register(
    id='ResourceMountainCar-v2',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:ResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 100
    },
)

register(
    id='DiscreteResourceMountainCar-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:DiscreteResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 10
    },
)

register(
    id='ContinuousMountainCar-v0',
    entry_point='mbrl.environments.our_envs.continuous_mountain_car:Continuous_MountainCarEnv',
    max_episode_steps=500,
)
######################################

# Racing Car
######################################
register(
    id='RacingCar-v0',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 19,
    },
    max_episode_steps=500,
)

register(
    id='RacingCar-v1',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 15,
    },
    max_episode_steps=500,
)

register(
    id='RacingCar-v2',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 10,
    },
    max_episode_steps=500,
)

register(
    id='RacingCar-v3',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 5,
    },
    max_episode_steps=500,
)

register(
    id='RacingCar-v4',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 18,
    },
    max_episode_steps=500,
)

register(
    id='RacingCar-v5',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:Racing',
    kwargs={
        "seed": None,
        "oil": 17.5,
    },
    max_episode_steps=500,
)

register(
    id='RacingCarSparse-v0',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:RacingSparseReward',
    kwargs={
        "seed": None,
        "oil": 19,
    },
    max_episode_steps=500,
)

register(
    id='RacingCarSparse-v1',
    entry_point='mbrl.environments.our_envs.racing_car.racing_car:RacingSparseReward',
    kwargs={
        "seed": None,
        "oil": 18,
    },
    max_episode_steps=500,
)
######################################

# Cheetah (resource)
######################################
register(
    id='MagellanHalfCheetah-v2',
    entry_point='mbrl.environments.our_envs.half_cheetah:MagellanHalfCheetahEnv',
    max_episode_steps=100
)

register(
    id='CheetahCorridor-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:CheetahCorridor',
    kwargs={
        'reward_block': [4,5],
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridor-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridorV2-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridorV2',
    kwargs={
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridor-v01',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridor',
    kwargs={
        'cargo_num': 8,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridor-v02',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridor',
    kwargs={
        'cargo_num': 10,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridor-v1',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [3,4],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='ResourceCheetahCorridor-v2',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:ResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [2,3],
        'reward': 100
    },
    max_episode_steps=500,
)
######################################

# Swimmer
######################################
register(
    id='SwimmerCorridor-v0',
    entry_point='mbrl.environments.our_envs.resource_swimmer.resource_swimmer_corridor:SwimmerCorridor',
    kwargs={
        'reward_block': [3,4]
    },
    max_episode_steps=500,
)
######################################



# Env with Done after exhausting resources
######################################
register(
    id='DoneResourceMountainCarEnv-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.resource_mountain_car:DoneResourceMountainCarEnv',
    max_episode_steps=500,
    kwargs={
        'seed': None,
        'cargo_num': 10
    },
)

register(
    id='DoneResourceCheetahCorridor-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:DoneResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='DoneResourceCheetahCorridor-v1',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:DoneResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [8,9],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='DoneResourceCheetahCorridor-v2',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:DoneResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [9,10],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='DoneResourceCheetahCorridor-v3',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:DoneResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [7,8],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='DoneResourceCheetahCorridor-v4',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:DoneResourceCheetahCorridor',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [6,7],
        'reward': 100
    },
    max_episode_steps=500,
)

register(
    id='DoneAntCorridorResourceEnv-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:DoneAntCorridorResourceEnv',
    kwargs={
        'cargo_num': 4,
        'beta': 5,
        'reward_block': [4,5],
        'reward': 100
    },
    max_episode_steps=500,
)
######################################

# ICML TO IJCAI Add experiments
register(
    id='FuelMountainCar-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 5.0
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCar-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 5.0,
        'alpha': 5
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCar-v2',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 5.0,
        'alpha': 10
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCar-v3',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 5.0,
        'alpha': 15
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCar-v31',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 4.0,
        'alpha': 15
    },
    max_episode_steps=200,
)
register(
    id='FuelMountainCar-v32',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarEnv',
    kwargs={
        'seed': None,
        'fuel_num': 6.0,
        'alpha': 15
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarR100-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarR100',
    kwargs={
        'seed': None,
        'fuel_num': 6.0,
        'alpha': 15
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarR100Done-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarR100Done',
    kwargs={
        'seed': None,
        'fuel_num': 6.0,
        'alpha': 15
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarR100Done-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarR100Done',
    kwargs={
        'seed': None,
        'fuel_num': 12.0,
        'alpha': 30
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarR100Done-v2',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarR100Done',
    kwargs={
        'seed': None,
        'fuel_num': 15.0,
        'alpha': 37.5
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarR100Done-v3',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarR100Done',
    kwargs={
        'seed': None,
        'fuel_num': 18.0,
        'alpha': 45
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarDone-v0',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarDone',
    kwargs={
        'seed': None,
        'fuel_num': 15.0,
        'alpha': 37.5
    },
    max_episode_steps=200,
)

register(
    id='FuelMountainCarDone-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel:FuelMountainCarDone',
    kwargs={
        'seed': None,
        'fuel_num': 12.0,
        'alpha': 30
    },
    max_episode_steps=200,
)

register(
    id='ContinuousMountainCar-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car:ContinuousMountainCarEnv',
    kwargs={
        'seed': None
    },
    max_episode_steps=200,
)

register(
    id='AntCorridorEnvActionCost-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [6,7]
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorEnvActionCost-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [5,6]
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorEnvActionCost-v2',
    entry_point='mbrl.environments.our_envs.ant_corridor:AntCorridorEnv',
    kwargs={
        'reward_block': [4,5]
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorActionCost-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.resource_cheetah_corridor:CheetahCorridor',
    kwargs={
        'reward_block': [6,7],
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuel-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuel',
    kwargs={
        'cargo_num': 20,
        'beta': 1,
        'reward_block': [6,7],
        'reward': 100,
        'alpha': 50 # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuel-v1',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuel',
    kwargs={
        'cargo_num': 15,
        'beta': 1,
        'reward_block': [6,7],
        'reward': 100,
        'alpha': 37.5  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuel-v2',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuel',
    kwargs={
        'cargo_num': 13,
        'beta': 1,
        'reward_block': [6,7],
        'reward': 100,
        'alpha': 35  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuel-v3',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuel',
    kwargs={
        'cargo_num': 14,
        'beta': 1,
        'reward_block': [6,7],
        'reward': 100,
        'alpha': 35  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuel-v4',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuel',
    kwargs={
        'cargo_num': 8,
        'beta': 1,
        'reward_block': [6,7],
        'reward': 100,
        'alpha': 20  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuelDone-v0',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuelDone',
    kwargs={
        'cargo_num': 36,
        'beta': 1,
        'reward_block': [9,10],
        'reward': 100,
        'alpha': 90.0  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuelDone-v1',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuelDone',
    kwargs={
        'cargo_num': 27,
        'beta': 1,
        'reward_block': [9,10],
        'reward': 100,
        'alpha': 67.5  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuelDone-v2',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuelDone',
    kwargs={
        'cargo_num': 30,
        'beta': 1,
        'reward_block': [9,10],
        'reward': 100,
        'alpha': 75  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='CheetahCorridorFuelDone-v3',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel:CheetahCorridorFuelDone',
    kwargs={
        'cargo_num': 32,
        'beta': 1,
        'reward_block': [9,10],
        'reward': 100,
        'alpha': 80  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuel-v0',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuel',
    kwargs={
        'cargo_num': 200,
        'beta': 1,
        'reward_block': [5,6],
        'reward': 100,
        'alpha': 500  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuel-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuel',
    kwargs={
        'cargo_num': 180,
        'beta': 1,
        'reward_block': [5,6],
        'reward': 100,
        'alpha': 450  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 360,
        'beta': 1,
        'reward_block': [5,6],
        'reward': 100,
        'alpha': 900  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v2',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 180,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 450  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v3',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 160,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 400  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v4',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 150,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 375  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v5',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 140,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 350  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuelDone-v50',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 140,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 280  # 2.5 * fuel
    },
    max_episode_steps=500,
)
register(
    id='AntCorridorFuelDone-v51',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuelDone',
    kwargs={
        'cargo_num': 140,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 420  # 2.5 * fuel
    },
    max_episode_steps=500,
)

register(
    id='AntCorridorFuel-v2',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel:AntCorridorFuel',
    kwargs={
        'cargo_num': 80,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': 200  # 2.5 * fuel
    },
    max_episode_steps=500,
)

### fuel cargo mountain car
register(
    id='FuelCargoMountainCar-v1',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel_cargo:FuelCargoMountainCar',
    kwargs={
        'seed': None,
        'fuel_num': 12.0,
        'cargo_num': 10,
        'alpha': [30, 1]
    },
    max_episode_steps=200,
)

register(
    id='FuelCargoMountainCar-v2',
    entry_point='mbrl.environments.our_envs.mountain_car.continuous_mountain_car_fuel_cargo:FuelCargoMountainCar',
    kwargs={
        'seed': None,
        'beta': 10,
        'fuel_num': 12.0,
        'cargo_num': 10,
        'alpha': [30, 1]
    },
    max_episode_steps=200,
)
### fuel cargo mountain car

### fuel cargo ant
register(
    id='AntCorridorFuelCargo-v1',
    entry_point='mbrl.environments.our_envs.ant_corridor_fuel_cargo:AntCorridorFuelCargo',
    kwargs={
        'fuel_num': 140,
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': [350, 1]  # 2.5 * fuel
    },
    max_episode_steps=500,
)
### fuel cargo ant

### fuel cargo cheetah
register(
    id='CheetahCorridorFuelCargo-v1',
    entry_point='mbrl.environments.our_envs.resource_cheetah.cheetah_corridor_fuel_cargo:CheetahCorridorFuelCargo',
    kwargs={
        'fuel_num': 32,
        'cargo_num': 4,
        'beta': 1,
        'reward_block': [4,5],
        'reward': 100,
        'alpha': [80, 1]
    },
    max_episode_steps=500,
)
### fuel cargo cheetah

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
    "mountain_car": "MountainCarContinuous-v0",
    "resource_mountaincar_v0": "ResourceMountainCar-v0",
    "resource_mountaincar_v1": "ResourceMountainCar-v1",
    "resource_mountaincar_v2": "ResourceMountainCar-v2",
    "resource_mountaincar_v3": "ResourceMountainCar-v3",
    "resource_mountaincar_v4": "ResourceMountainCar-v4",
    "resource_mountaincar_v5": "ResourceMountainCar-v5",
    "resource_mountaincar_v6": "ResourceMountainCar-v6",
    "resource_mountaincar_v7": "ResourceMountainCar-v7",
    "resource_mountaincar_v8": "ResourceMountainCar-v8",
    "mountain_car_v1": "ContinuousMountainCar-v0",
    "racing_car_v0": "RacingCar-v0",
    "racing_car_v1": "RacingCar-v1",
    "racing_car_v2": "RacingCar-v2",
    "racing_car_v3": "RacingCar-v3",
    "racing_car_v4": "RacingCar-v4",
    "racing_car_v5": "RacingCar-v5",
    "racing_car_sparse_v0": "RacingCarSparse-v0",
    "racing_car_sparse_v1": "RacingCarSparse-v1",
    "discrete_resource_mountaincar_v0": "DiscreteResourceMountainCar-v0",
    "ant_maze": "MagellanAnt-v2",
    "cheetah_sparse": "MagellanHalfCheetah-v2",
    "n_chain":"NChainOur-v0",
    "ant_maze_resource": "AntMazeResource-v0",
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
    'reward_ant_corridor_resource_env_v0': 'RewardAntCorridorResourceEnv-v0',
    'ant_corridor_env_3': "AntCorridorEnv-v3",
    'ant_corridor_env_4': "AntCorridorEnv-v4",
    "ant_corridor_env_5": "AntCorridorEnv-v5",
    "ant_corridor_env_6": "AntCorridorEnv-v6",
    "cheetah_corridor_v0": "CheetahCorridor-v0",
    "resource_cheetah_corridor_v0": "ResourceCheetahCorridor-v0",
    "resource_cheetah_corridor_v01": "ResourceCheetahCorridor-v01",
    "resource_cheetah_corridor_v02": "ResourceCheetahCorridor-v02",
    "resource_cheetah_corridor_v1": "ResourceCheetahCorridor-v1",
    "resource_cheetah_corridor_v2": "ResourceCheetahCorridor-v2",
    "swimmer_corridor_v0": "SwimmerCorridor-v0",
    "no_reward_resource_ant_corridor": "NoRewardAntCorridorResourceEnv-v0",
    "done_resource_mountaincar_v0": "DoneResourceMountainCarEnv-v0",
    "done_resource_cheetah_corridor_v0": "DoneResourceCheetahCorridor-v0",
    "done_ant_corridor_resource_env_goal_4_v0": "DoneAntCorridorResourceEnv-v0",
    "done_resource_cheetah_corridor_v1": "DoneResourceCheetahCorridor-v1",
    "done_resource_cheetah_corridor_v2": "DoneResourceCheetahCorridor-v2",
    "done_resource_cheetah_corridor_v3": "DoneResourceCheetahCorridor-v3",
    "done_resource_cheetah_corridor_v4": "DoneResourceCheetahCorridor-v4",
    "no_reward_resource_mountaincar": "NoRewardResourceMountainCar-v0",
    "ant_corridor_resource_env_goal_4_v0_v2": "AntCorridorResourceEnvV2-v0",
    "ant_corridor_resource_env_goal_4_v0_v2_2cargo": "AntCorridorResourceEnvV2cargo2-v0",
    "ant_corridor_resource_env_goal_4_v0_v2_8cargo": "AntCorridorResourceEnvV2cargo8-v0",
    "ant_corridor_resource_env_goal_4_v0_v2_16cargo": "AntCorridorResourceEnvV2cargo16-v0",
    "resource_cheetah_corridor_v0_v2": "ResourceCheetahCorridorV2-v0",
    "resource_mountaincar_v0_v2": "ResourceMountainCarV2-v0",
    "fuel_mountain_car_v0": 'FuelMountainCar-v0',
    "fuel_mountain_car_v1": 'FuelMountainCar-v1',
    "fuel_mountain_car_v2": 'FuelMountainCar-v2',
    "fuel_mountain_car_v3": 'FuelMountainCar-v3',
    "fuel_mountain_car_v31": 'FuelMountainCar-v31',
    "fuel_mountain_car_v32": 'FuelMountainCar-v32',
    "fuel_mountain_car_r100_v0": "FuelMountainCarR100-v0",
    "fuel_mountain_car_r100_done_v0": "FuelMountainCarR100Done-v0",
    "fuel_mountain_car_r100_done_fuel12_v1": "FuelMountainCarR100Done-v1",
    "fuel_mountain_car_r100_done_fuel15_v2": "FuelMountainCarR100Done-v2",
    "fuel_mountain_car_r100_done_fuel18_v3": "FuelMountainCarR100Done-v3",
    "fuel_mountain_car_done_fuel15_v0": "FuelMountainCarDone-v0",
    "fuel_mountain_car_done_fuel12_v1": "FuelMountainCarDone-v1",
    "continuous_mountaincar_action_cost": "ContinuousMountainCar-v1",
    "ant_corridor_action_cost_v0": "AntCorridorEnvActionCost-v0",
    "ant_corridor_action_cost_v1": "AntCorridorEnvActionCost-v1",
    "ant_corridor_action_cost_v2": "AntCorridorEnvActionCost-v2",
    "cheetah_corridor_action_cost_v0": "CheetahCorridorActionCost-v0",
    "cheetah_corridor_fuel_v0": "CheetahCorridorFuel-v0",
    "cheetah_corridor_fuel_15_v1": "CheetahCorridorFuel-v1",
    "cheetah_corridor_fuel_13_v2": "CheetahCorridorFuel-v2",
    "cheetah_corridor_fuel_14_v3": "CheetahCorridorFuel-v3",
    "cheetah_corridor_fuel_8_v4": "CheetahCorridorFuel-v4",
    "ant_corridor_fuel_200_v0": "AntCorridorFuel-v0",
    "ant_corridor_fuel_180_v1": "AntCorridorFuel-v1",
    "ant_corridor_fuel_done_360_v1": "AntCorridorFuelDone-v1",
    "ant_corridor_fuel_80_goal4_v2": "AntCorridorFuel-v2",
    "ant_corridor_fuel_done_180_v2": "AntCorridorFuelDone-v2",
    "ant_corridor_fuel_done_160_v3": "AntCorridorFuelDone-v3",
    "ant_corridor_fuel_done_150_v4": "AntCorridorFuelDone-v4",
    "ant_corridor_fuel_done_140_v5": "AntCorridorFuelDone-v5",
    "cheetah_corridor_fuel_done_36_goal9_v0": "CheetahCorridorFuelDone-v0",
    "cheetah_corridor_fuel_done_27_goal9_v1": "CheetahCorridorFuelDone-v1",
    "cheetah_corridor_fuel_done_30_goal9_v2": "CheetahCorridorFuelDone-v2",
    "cheetah_corridor_fuel_done_32_goal9_v3": "CheetahCorridorFuelDone-v3",
    "ant_corridor_resource_env_goal_4_beta1_alpha05": "AntCorridorResourceEnvV2-v1",
    "ant_corridor_resource_env_goal_4_beta1_alpha2": "AntCorridorResourceEnvV2-v2",
    "ant_corridor_resource_env_goal_4_beta1_alpha15": "AntCorridorResourceEnvV2-v3",
    "ant_corridor_fuel_done_140_beta1_alpha2fuel": "AntCorridorFuelDone-v50",
    "ant_corridor_fuel_done_140_beta1_alpha3fuel": "AntCorridorFuelDone-v51",
    "fuel_cargo_car_v1": "FuelCargoMountainCar-v1",
    "fuel_cargo_car_v2": "FuelCargoMountainCar-v2",
    "ant_fuel_cargo_v1": "AntCorridorFuelCargo-v1",
    "cheetah_fuel_cargo_v1": "CheetahCorridorFuelCargo-v1",
    "ant_goal4_cargo_resource_beta1": "AntCorridorResourceEnvBeta1-v5"
}
