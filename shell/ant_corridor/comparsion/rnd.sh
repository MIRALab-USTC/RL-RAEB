#!/bin/bash
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal42.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal43.txt 2>&1 &
