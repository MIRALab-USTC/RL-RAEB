#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_less.json > o_car_less1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_less.json > o_car_less2.txt 2>&1 &
sleep 10s 
CUDA_VISIBLE_DEVICES=2 nohup  python scripts/run.py configs/model-based/modelbased_sac_mountaincar_model_less.json > o_car_less3.txt 2>&1 &

