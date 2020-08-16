#!/bin/bash
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_human.json > o_human_virtualloss1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_human.json > o_human_virtualloss2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_human.json > o_human_virtualloss3.txt 2>&1 &