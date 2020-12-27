#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise_costs/src_goal5.json > src_goal51.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise_costs/src_goal5.json > src_goal52.txt 2>&1 &
sleep 15s 

#CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal5.json > rnd_ant_corridor_goal51.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal5.json > rnd_ant_corridor_goal52.txt 2>&1 &
#sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal5.json > surprise_ant_corridor_goal51.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal5.json > surprise_ant_corridor_goal52.txt 2>&1 &
sleep 15s 
#CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal5.json > rnd_vision_ant_corridor_goal51.txt 2>&1 &
#sleep 15s 
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json > surprise_ant_corridor_goal4.txt 2>&1 &
