#!/bin/bash
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5_int005_seed10.json > surprise_vision_ant_corridor_goal5_int005_seed10.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5_int005_seed100.json > surprise_vision_ant_corridor_goal5_int005_seed100.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4_int005_seed10.json > surprise_vision_ant_corridor_goal4_int005_seed10.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4_int005_seed100.json > surprise_vision_ant_corridor_goal4_int005_seed100.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4_int005_seed1000.json > surprise_vision_ant_corridor_goal4_int005_seed1000.txt 2>&1 &

