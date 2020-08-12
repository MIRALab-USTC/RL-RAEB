#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_ant_maze.json > o_ant_Surprise_max_len_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_ant_maze.json > o_ant_Surprise_max_len_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_ant_maze.json > o_ant_Surprise_max_len_3.txt 2>&1 &
