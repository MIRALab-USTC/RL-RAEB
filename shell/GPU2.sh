#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_ant.json > o_ant_max_state_entropyk11_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_ant.json > o_ant_max_state_entropyk11_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_ant.json > o_ant_max_state_entropyk11_3.txt 2>&1 &
