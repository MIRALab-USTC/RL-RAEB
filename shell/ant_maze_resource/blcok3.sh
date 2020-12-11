#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta10.json > sac_surprise_vision_ant_maze_resource_block3_beta101.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta10.json > sac_surprise_vision_ant_maze_resource_block3_beta102.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta50.json > sac_surprise_vision_ant_maze_resource_block3_beta501.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta50.json > sac_surprise_vision_ant_maze_resource_block3_beta502.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta100.json > sac_surprise_vision_ant_maze_resource_block3_beta1001.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/Block3/sac_surprise_vision_ant_maze_resource_block3_beta100.json > sac_surprise_vision_ant_maze_resource_block3_beta1002.txt 2>&1 &
