#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_k_32.json > o_maze_hash_state_action_k_32.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_k_64.json > o_maze_hash_state_action_k_64.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_k_128.json > o_maze_hash_state_action_k_128.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_k_256.json > o_maze_hash_state_action_k_256.txt 2>&1 &


