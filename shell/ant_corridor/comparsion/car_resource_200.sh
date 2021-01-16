#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 2  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v7  --repeat 4 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise > surprise_100_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v7  --repeat 5 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_2002.txt 2>&1 &


#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v4  --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise > surprise_5_2.txt 2>&1 &
#sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise > surprise_10_2.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 2  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v4 --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_5/surprise_vision_shape_weight > surprise_vision_shape_weight_5_2.txt 2>&1 &
#sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_10/surprise_vision_shape_weight > surprise_vision_shape_weight_10_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v8 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_15/surprise > surprise_15_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v8 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_15/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_2001.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v8  --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_15/surprise > surprise_15_2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v8 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_15/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_2002.txt 2>&1 &











