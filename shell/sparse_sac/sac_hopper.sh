#!/bin/bash
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/sac/sac_hopper.json > o_hopper_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac/sac_hopper.json > o_hopper_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac/sac_hopper.json > o_hopper_sac3.txt 2>&1 &
