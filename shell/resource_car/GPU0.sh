#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/surprise_resource_car.json > surprise_resource_car1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/surprise_resource_car.json > surprise_resource_car2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/surprise_resource_car.json > surprise_resource_car3.txt 2>&1 &
