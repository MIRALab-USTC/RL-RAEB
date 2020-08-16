#!/bin/bash
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_human.json > o_MSE_human1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_human.json > o_MSE_human2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_human.json > o_MSE_human3.txt 2>&1 &
