#!/bin/bash
#CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_hopper.json > o_hopper_max_state_entropyk11_1.txt 2>&1 &
#sleep 10s
#CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_hopper.json > o_hopper_max_state_entropyk11_2.txt 2>&1 &
#sleep 10s
#CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_hopper.json > o_hopper_max_state_entropyk11_3.txt 2>&1 &
#sleep 10s

CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_hopper.json > o_hopper_surprise11_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_hopper.json > o_hopper_surprise11_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/modelbased_sac_surprise_hopper.json > o_hopper_surprise11_3.txt 2>&1 &
sleep 10s