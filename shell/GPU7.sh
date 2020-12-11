#!/bin/bash
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_car.json > o_car_mse1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_car.json > o_car_mse2.txt 2>&1 &
