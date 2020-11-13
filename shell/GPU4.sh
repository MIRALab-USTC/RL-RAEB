#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker2d_max_state_entropyk11_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker2d_max_state_entropyk11_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based-max-state_entropy/modelbased_sac_virtual_reward_state_entropy_surprise_walker2d.json > o_walker2d_max_state_entropyk11_3.txt 2>&1 &



python scripts/run.py configs/sac/sac_hash/sac_with_hash.json