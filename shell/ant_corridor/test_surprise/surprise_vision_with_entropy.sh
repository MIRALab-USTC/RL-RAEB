#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5.json > surprise_vision_ant_corridor_goal51.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5.json > surprise_vision_ant_corridor_goal52.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4.json > surprise_vision_ant_corridor_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4.json > surprise_vision_ant_corridor_goal42.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal6.json > surprise_vision_ant_corridor_goal61.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal6.json > surprise_vision_ant_corridor_goal62.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal7.json > surprise_vision_ant_corridor_goal71.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal7.json > surprise_vision_ant_corridor_goal72.txt 2>&1 &