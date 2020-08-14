#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_swimmer.json > o_Surprise_swimmer1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_swimmer.json > o_Surprise_swimmer2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_swimmer.json > o_Surprise_swimmer2.txt 2>&1 &
