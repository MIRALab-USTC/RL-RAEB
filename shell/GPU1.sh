#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json >  o_car_max_state_entropy.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json > o_maze_max2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json > o_maze_max3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_kde_surprise_car.json


CUDA_VISIBLE_DEVICES=0 python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_car.json > o_test_car_new_idea.txt 2>&1 &
