#!/bin/bash
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal5.json > rnd_vision_ant_corridor_goal55.txt 2>&1 &