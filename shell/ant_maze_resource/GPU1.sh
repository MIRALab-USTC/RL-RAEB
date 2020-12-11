#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash_stateaction_cnt_sqrt/sac_with_hash_state_action_k_16_maze_resource.json > sac_with_hash_state_action_k_16_maze_resource1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash_stateaction_cnt_sqrt/sac_with_hash_state_action_k_16_maze_resource.json > sac_with_hash_state_action_k_16_maze_resource2.txt 2>&1 &

