#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_resource_car.json > sac_resource_car1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_resource_car.json > sac_resource_car2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_resource_car.json > sac_resource_car3.txt 2>&1 &
