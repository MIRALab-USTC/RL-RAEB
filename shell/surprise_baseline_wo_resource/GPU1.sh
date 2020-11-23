#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_sqrt_k16.json > o_maze_hash_sqrt_state_k_16.txt 2>&1 &
#sleep 10s
#CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_sqrt_k32.json > o_maze_hash_sqrt_state_k_32.txt 2>&1 &
#sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_s_x_y_tabular/sac_with_tabular.json > o_next_state.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_ant_maze/sac_.json > o_sac.txt 2>&1 &

