#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_freq.json > o_car_freq1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_freq.json > o_car_freq2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_freq.json > o_car_freq3.txt 2>&1 &

