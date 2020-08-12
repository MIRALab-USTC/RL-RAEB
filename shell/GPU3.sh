#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_trainfreq1.json > o_car_Surprise_train_model_freq1_maxlen200.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_trainfreq1.json > o_car_Surprise_train_model_freq2_maxlen200.txt 2>&1 &


