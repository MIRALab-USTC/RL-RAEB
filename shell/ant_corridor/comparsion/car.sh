#!/bin/bash
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/discrete_resource_10/sac > sac_discrete_resource_101.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/discrete_resource_10/sac > sac_discrete_resource_102.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v3 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/discrete_resource_25/sac > sac_discrete_resource_251.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v3 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/discrete_resource_25/sac > sac_discrete_resource_252.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/sac > sac1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/sac > sac1.txt 2>&1 &
# backup 
#sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/surprise > surprise1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/surprise > surprise2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v3   --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25/sac > surprise3.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v3   --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_25/sac > surprise4.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_10/surprise > surprise5.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_10/surprise > surprise6.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight3.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/mountain_car_test/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight4.txt 2>&1 &




