#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json > o_maze_max1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json > o_maze_max2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/max/modelbased_sac_max_ant_maze.json > o_maze_max3.txt 2>&1 &

