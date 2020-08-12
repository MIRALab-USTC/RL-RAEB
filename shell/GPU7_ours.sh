#!/bin/bash
for i in $(seq 1 3)
do
    CUDA_VISIBLE_DEVICES=7 python scripts/run.py configs/surprise-virtual-novelty/modelbased_sac_virtual_loss_surprise_swimmer.json
done



