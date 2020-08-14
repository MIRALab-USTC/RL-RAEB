#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_swimmer.json > o_swimmer_virtualloss_1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_swimmer.json > o_swimmer_virtualloss_2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_swimmer.json > o_swimmer_virtualloss_3.txt 2>&1 &
