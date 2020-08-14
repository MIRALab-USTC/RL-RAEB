#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_human.json > o_Surprise_human.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_human.json > o_Surprise_human.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_human.json > o_Surprise_human.txt 2>&1 &
