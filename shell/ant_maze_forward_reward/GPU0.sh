#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_.json > o_sac_forward_ant_maze1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_goal_forward_reward.json > sac_goal_forward_reward.txt 2>&1 &

