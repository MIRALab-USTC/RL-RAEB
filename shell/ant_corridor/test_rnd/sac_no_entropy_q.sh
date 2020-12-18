#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal3_no_resource.json > sac_goal3_no_resource1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal3_no_resource.json > sac_goal3_no_resource2.txt 2>&1 &