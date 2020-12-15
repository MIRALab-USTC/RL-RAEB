#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json > sac_goal61.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json > sac_goal62.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal6.json > surprise_ant_corridor_goal61.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal6.json > surprise_ant_corridor_goal62.txt 2>&1 &
#sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal7.json > surprise_vision_ant_corridor_goal71.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal7.json > surprise_vision_ant_corridor_goal72.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal6.json > surprise_vision_ant_corridor_goal61.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal6.json > surprise_vision_ant_corridor_goal62.txt 2>&1 &