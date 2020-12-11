#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/beta001_gamma999_forward_reward_maze/sac_with_hash_sqrt_k16_ant_maze_block2.json > sac_with_hash_sqrt_k16_ant_maze_block21.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_ant_maze_block2.json > sac_ant_maze_block2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_ant_maze.json > sac_surprise_ant_maze_block2.txt 2>&1 &
