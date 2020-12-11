#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_ant_maze_block4.json > sac_ant_maze_block41.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/sac_surprise_ant_maze_resource_block4.json > sac_surprise_ant_maze_resource_block41.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/sac_surprise_vision_ant_maze_resource_block4.json > sac_surprise_vision_ant_maze_resource_block41.txt 2>&1 &
