#!/bin/bash
for i in $(seq 1 3)
do
    CUDA_VISIBLE_DEVICES=7 python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_swimmer.json
done
