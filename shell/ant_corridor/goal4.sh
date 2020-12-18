#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_3.json > rnd_ant_corridor_31.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=0 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_3.json > rnd_ant_corridor_32.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_4.json > rnd_ant_corridor_41.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_4.json > rnd_ant_corridor_42.txt 2>&1 &

sleep 15s
CUDA_VISIBLE_DEVICES=2 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal41.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal4.json > rnd_ant_corridor_goal42.txt 2>&1 &

sleep 15s
CUDA_VISIBLE_DEVICES=3 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal5.json > rnd_ant_corridor_goal51.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal5.json > rnd_ant_corridor_goal52.txt 2>&1 &

sleep 15s
CUDA_VISIBLE_DEVICES=4 python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal41.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal4.json > rnd_vision_ant_corridor_goal42.txt 2>&1 &

sleep 15s
CUDA_VISIBLE_DEVICES=5 python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal5.json > rnd_vision_ant_corridor_goal51.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal5.json > rnd_vision_ant_corridor_goal52.txt 2>&1 &