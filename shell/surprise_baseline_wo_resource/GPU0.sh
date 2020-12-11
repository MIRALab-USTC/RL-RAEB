#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_hash_stateaction_cnt_sqrt/sac_with_hash_state_action_k_16.json > o_maze_hash_sqrt_k_10.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_hash_stateaction_cnt_sqrt/sac_with_hash_state_action_k_32.json > o_maze_hash_sqrt_k_32.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_fix_bug_k16.json > o_maze_hash_state_no_sqrt_k_16.txt 2>&1 &


CUDA_VISIBLE_DEVICES=2 python scripts/run.py configs/sac/sac_hash/sac_with_hash_sqrt_k32.json

CUDA_VISIBLE_DEVICES=2 python scripts/run.py configs/sac/sac_s_x_y/sac_with_hash_sqrt_k8.json


CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_s_x_y_tabular/sac_with_tabular.json o_next_state.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_.json o_sac.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/resource-augmented/comparsion/modelbased_sac_surprise_ant_maze.json > o_maze_surprise.txt 2>&1 &
