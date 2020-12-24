#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_costs/src_goal4.json > src_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_costs/src_goal4.json > src_goal42.txt 2>&1 &
sleep 15s 

CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal42.txt 2>&1 &
sleep 15s 

CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal43.txt 2>&1 &
