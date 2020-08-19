#!/bin/bash
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac3.txt 2>&1 &
