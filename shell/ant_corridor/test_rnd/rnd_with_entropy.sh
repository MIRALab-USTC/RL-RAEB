#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_3_with_entropy.json > rnd_ant_corridor_3_with_entropy3.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_3_with_entropy.json > rnd_ant_corridor_3_with_entropy4.txt 2>&1 &
