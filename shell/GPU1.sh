#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_car.json > o_car_max_state_entropyk17_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_car.json > o_car_max_state_entropyk17_2.txt 2>&1 &

