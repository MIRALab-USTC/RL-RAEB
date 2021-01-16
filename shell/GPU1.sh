#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_k_10.json > o_maze_hash_k_10.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac_hash/sac_with_hash_k_32.json > o_maze_hash_k_32.txt 2>&1 &
sleep 10s

CUDA_VISIBLE_DEVICES=0 python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal6.json


CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal6.json > nohup_log/rnd_ant_corridor_goal6_cuda0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal6.json > nohup_log/rnd_ant_corridor_goal6_cuda1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal6.json > nohup_log/rnd_ant_corridor_goal6_cuda2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd/ant_corridor/rnd_ant_corridor_goal6.json > nohup_log/rnd_ant_corridor_goal6_cuda3.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal6_tx.json > nohup_log/surprise_vision/surprise_vision_ant_corridor_goal6_cuda0_2020_12_27_22_04.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json > nohup_log/sac/sac_ant_corridor_goal6_cuda0_2020_12_27_22_22.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal6_tx.json > nohup_log/surprise/surprise_ant_corridor_goal6_cuda0_2020_12_27_22_47.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/ant_maze_resource/surprise/ant_corridor/surprise_ant_corridor_goal6_tx.json > nohup_log/surprise/surprise_ant_corridor_goal6_cuda1_2020_12_28_21_01.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal6_tx.json > nohup_log/rnd_vision/rnd_vision_ant_corridor_goal6_cuda1_2020_12_28_21_10.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/rnd/ant_maze_resource/rnd_vision/ant_corridor/rnd_vision_ant_corridor_goal6_tx.json > nohup_log/rnd_vision/rnd_vision_ant_corridor_goal6_cuda1_2020_12_28_21_16.txt 2>&1 &