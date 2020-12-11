#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/beta001_gamma999_forward_reward_maze/sac_with_hash_sqrt_k16.json > o_sac_hash_goal_ant_maze1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/beta001_gamma999_forward_reward_maze/sac_with_hash_sqrt_k16_goal_forward_reward.json > sac_with_hash_sqrt_k16_goal_forward_reward.txt 2>&1 &

