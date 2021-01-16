#!/bin/bash
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/sac > sac_racing_car_v011.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/sac > sac_racing_car_v022.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_no_video_env.json  --env_name racing_car_v5 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_normalize_reward_1/racing_car_v5/sac > sac_racing_car_v111.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_no_video_env.json  --env_name racing_car_v5 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_normalize_reward_1/racing_car_v5/sac > sac_racing_car_v122.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/surprise > surprise1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/surprise > surprise2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_no_video_env.json  --env_name racing_car_v5 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_normalize_reward_1/racing_car_v5/surprise > surprise3.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_no_video_env.json  --env_name racing_car_v5 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_normalize_reward_1/racing_car_v5/surprise > surprise4.txt 2>&1 &
#sleep 15s

#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/surprise_vision_shape_weight_int005 > surprise_vision_shape_weight_sparse1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_sparse_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car_sparse/racing_car_sparse_v0/surprise_vision_shape_weight_int005 > surprise_vision_shape_weight_sparse2.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_v1 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car/racing_car_v1/surprise_vision_shape_weight > surprise_vision_shape_weight3continuous_resource_4.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_v1 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car/racing_car_v1/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_5.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_v2 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car/racing_car_v2/surprise_vision_shape_weight > surprise_vision_shape_weight3continuous_resource_6.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_no_video_env.json  --env_name racing_car_v2 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/racing_car/racing_car_v2/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_7.txt 2>&1 &
