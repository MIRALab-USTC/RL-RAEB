#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_no_sqrt_k_16.json > o_maze_hash_no_sqrt_state_k_16.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_hash_stateaction/sac_with_hash_state_action_no_sqrt_k_32.json > o_maze_hash_no_sqrt_state_k_32.txt 2>&1 &
sleep 10s