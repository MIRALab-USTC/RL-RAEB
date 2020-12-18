#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal3_no_resource.json > sac_goal3_no_resource1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal3_no_resource.json > sac_goal3_no_resource2.txt 2>&1 &
sleep 15s


CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal5.json > surprise_ant_corridor_goal51.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal5.json > surprise_ant_corridor_goal52.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5.json > surprise_vision_ant_corridor_goal51.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal5.json > surprise_vision_ant_corridor_goal52.txt 2>&1 &