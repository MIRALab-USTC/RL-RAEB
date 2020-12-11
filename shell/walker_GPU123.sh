#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker_max_state_entropyk11_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker_max_state_entropyk11_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker_max_state_entropyk11_3.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_walker.json > o_walker_surprise11_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_walker.json > o_walker_surprise11_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_walker.json > o_walker_surprise11_3.txt 2>&1 &
sleep 10s