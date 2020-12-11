#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/sac_surprise_ant_maze_resource.json > sac_surprise_ant_maze_resource1.txt 2>&1 &
sleep 10s 
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/sac_surprise_ant_maze_resource.json > sac_surprise_ant_maze_resource2.txt 2>&1 &