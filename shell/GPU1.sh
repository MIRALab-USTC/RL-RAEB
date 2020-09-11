#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/surprise_ant_maze_resource.json > o_maze_resource_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/surprise_ant_maze_resource.json > o_maze_resource_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/surprise_ant_maze_resource.json > o_maze_resource_3.txt 2>&1 &
