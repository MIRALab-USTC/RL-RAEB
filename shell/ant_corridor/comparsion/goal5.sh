#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal4.json > sac_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal4.json > sac_goal42.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal4.json > sac_goal43.txt 2>&1 &
sleep 15s 
#CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal41.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal42.txt 2>&1 &
#sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4_int005.json > surprise_vision_ant_corridor_goal4_int0051.txt 2>&1 &
sleep 15s 
#CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal41.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal42.txt 2>&1 &

