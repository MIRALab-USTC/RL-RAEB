#!/bin/bash
#CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/rnd/rnd_vision.json > rnd_vision1.txt 2>&1 &
#sleep 15s 
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/rnd/rnd_vision.json > rnd_vision2.txt 2>&1 &
sleep 15s 

CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/rnd_vision.json > rnd_vision3.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/rnd_vision.json > rnd_vision4.txt 2>&1 &
sleep 15s 

CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/rnd/rnd_vision.json > rnd_vision5.txt 2>&1 &

