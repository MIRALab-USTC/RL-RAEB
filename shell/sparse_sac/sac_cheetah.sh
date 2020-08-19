#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac3.txt 2>&1 &
