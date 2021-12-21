#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision_small_model_fuel_car_alpha15_fuel6.json > surprise_vision_small_model_fuel_car_alpha15_fuel61.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision_small_model_fuel_car_alpha15_fuel6.json > surprise_vision_small_model_fuel_car_alpha15_fuel62.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_action_cost_v1 --repeat 1 > surprise_ant_action_cost.txt 2>&1 &
# sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name cheetah_corridor_fuel_8_v4  > surprise_vision_cheetah_corridor_fuel_8_v4.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_corridor_fuel_8_v4   --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/envs_fuel/surprise > surprise_cheetah_corridor_fuel_8_v4.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_action_cost_v2   --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/envs_action_cost/surprise > ant_corridor_action_cost_v2.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_small_model_car_action_cost.json > surprise_small_model_car_action_cost.txt 2>&1 &