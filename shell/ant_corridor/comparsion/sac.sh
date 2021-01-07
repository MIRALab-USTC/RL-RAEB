#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v6  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_200/sac_path200 > sac_path200_continuous_resource_2001.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v6  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_200/sac_path200 > sac_path200_continuous_resource_2002.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v5  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_2/sac_path200 > sac_path200_continuous_resource_21.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v5  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_2/sac_path200 > sac_path200_continuous_resource_22.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v4  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/sac_path200 > sac_path200_continuous_resource_51.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v4  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/sac_path200 > sac_path200_continuous_resource_52.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25/sac_path200 > sac_path200_continuous_resource_251.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25/sac_path200 > sac_path200_continuous_resource_252.txt 2>&1 &


