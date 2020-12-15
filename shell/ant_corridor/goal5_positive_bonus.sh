#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5_positive_bonus.json > surprise_vision_ant_corridor_goal5_positive_bonus1.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5_positive_bonus.json > surprise_vision_ant_corridor_goal5_positive_bonus2.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5_positive_bonus.json > surprise_vision_ant_corridor_goal5_positive_bonus3.txt 2>&1 &

