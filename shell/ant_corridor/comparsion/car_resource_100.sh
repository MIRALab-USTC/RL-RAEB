#!/bin/bash
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v7   --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_100/sac_path200 > sac_path200_continuous_resource_1001.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v7   --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_100/sac_path200 > sac_path200_continuous_resource_1002.txt 2>&1 &
#sleep 15s
CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25_var_shape/surprise > surprise5continuous_resource_1001.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5_var_shape/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_11.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v3 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25_var_shape/surprise_vision_shape_weight > surprise_vision_shape_weight3continuous_resource_1001.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5_var_shape/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_1002.txt 2>&1 &


CUDA_VISIBLE_DEVICES=1 xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25_var_shape_test_bug/surprise

