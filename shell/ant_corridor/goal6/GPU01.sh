#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json > sac_goal61.txt 2>&1 &
#sleep 15s 
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json > sac_goal62.txt 2>&1 &
sleep 15s 

CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal6.json > rnd_vision_ant_corridor_goal61.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal6.json > rnd_vision_ant_corridor_goal62.txt 2>&1 &


