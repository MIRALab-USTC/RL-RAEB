#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_cheetah.json > o_cheetah_sac3.txt 2>&1 &

sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac/sac_walker.json > o_walker_sac3.txt 2>&1 &
