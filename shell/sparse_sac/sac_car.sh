#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_car.json > o_car_sac1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_car.json > o_car_sac2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_car.json > o_car_sac3.txt 2>&1 &
