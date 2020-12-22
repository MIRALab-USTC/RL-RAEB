#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_costs/src_goal4.json > src_goal4.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_costs/src_goal5.json > src_goal5.txt 2>&1 &

