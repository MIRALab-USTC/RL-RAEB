#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_k_10.json > o_maze_hash_k_10.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_k_32.json > o_maze_hash_k_32.txt 2>&1 &
sleep 10s


