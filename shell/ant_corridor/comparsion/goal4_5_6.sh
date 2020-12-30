#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --discount 0.999 > surprise_ant_corridor_goal41.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --discount 0.999 > surprise_ant_corridor_goal42.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --discount 0.995 > surprise_ant_corridor_goal43.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --discount 0.995 > surprise_ant_corridor_goal44.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json  --intrinsic_coeff 0.1 --max_step 50000 > surprise_ant_corridor_goal45.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json  --intrinsic_coeff 0.1 --max_step 50000 > surprise_ant_corridor_goal46.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json  --intrinsic_coeff 0.1 --max_step 100000 > surprise_ant_corridor_goal47.txt 2>&1 &
#sleep 15s 
#CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json  --intrinsic_coeff 0.1 --max_step 100000 > surprise_ant_corridor_goal48.txt 2>&1 &

