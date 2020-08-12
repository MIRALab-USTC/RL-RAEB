#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup  python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_cheetah.json > o_cheetah1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup  python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_cheetah.json > o_cheetah2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup  python scripts/run.py configs/surprise-based/modelbased_sac_surprise_car_cheetah.json > o_cheetah3.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=5 nohup  python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_cheetah.json > o_cheetah_ours1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup  python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_cheetah.json > o_cheetah_ours2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup  python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_cheetah.json > o_cheetah_ours3.txt 2>&1 &


