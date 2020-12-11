#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Best_20201210/sac_surprise_vision_ant_maze_resource_block24.json > sac_surprise_vision_ant_maze_resource_block241.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Best_20201210/sac_surprise_vision_ant_maze_resource_block24.json > sac_surprise_vision_ant_maze_resource_block242.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Best_20201210/sac_surprise_vision_ant_maze_resource_block24.json > sac_surprise_vision_ant_maze_resource_block242.txt 2>&1 &