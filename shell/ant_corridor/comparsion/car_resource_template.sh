#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --repeat 2 --env_name resource_mountaincar_v4  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_small_model_train_freq_low > surprise_small_model_train_freq_low1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_small_model_train_freq_low > surprise_small_model_train_freq_low2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json  --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_small_model_train_freq_low > surprise_small_model_train_freq_low3.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_vision_small_model_train_freq_low > surprise_vision_small_model_train_freq_low4.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_vision_small_model_train_freq_low > surprise_vision_small_model_train_freq_low5.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json  --env_name resource_mountaincar_v4  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_5/surprise_vision_small_model_train_freq_low > surprise_vision_small_model_train_freq_low1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --repeat 2 --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_25/surprise_small_model_train_freq_low > surprise_small_model_train_freq_low25_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --repeat 3 --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_25/surprise_small_model_train_freq_low > surprise_small_model_train_freq_low25_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json  --repeat 2 --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_25/surprise_vision_small_model_train_freq_low > surprise_vision_small_model_train_freq_low25_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json --repeat 3 --env_name resource_mountaincar_v3  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_small_model/continuous_resource_25/surprise_vision_small_model_train_freq_low > surprise_vision_small_model_train_freq_low25_2.txt 2>&1 &

:<<!
CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 2  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v7  --repeat 4 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise > surprise_100_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v7  --repeat 5 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_2002.txt 2>&1 &
!









