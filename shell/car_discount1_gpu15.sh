#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car.json > o_car_surprise_discount1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car.json > o_car_surprise_discount2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car.json > o_car_surprise_discount3.txt 2>&1 &
