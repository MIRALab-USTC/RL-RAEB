#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup  xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor/goal4/surprise > surprise1.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor/goal4/surprise_vision > surprise_ant_corridor_goal1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor/goal4/surprise_vision > surprise_ant_corridor_goal2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor/goal4/surprise > surprise2.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/cheetah/corridor/goal4/sac > sac1.txt 2>&1 &

:<<!
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/with_entropy_int001/goal4/surprise_vision > surprise_ant_corridor_goal43.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --base_log_dir /home/zhwang/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/with_entropy_int001/goal4/surprise_vision > surprise_ant_corridor_goal44.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json  --env_name resource_mountaincar_v7  --repeat 4 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise > surprise_100_1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json  --env_name resource_mountaincar_v7  --repeat 5 --base_log_dir /home/zhwang/research/ICML_data/mountain_car_test_fix_bug/continuous_resource_100/surprise_vision_shape_weight > surprise_vision_shape_weight4continuous_resource_2002.txt 2>&1 &

## 涛星已经跑起来的实验

#CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise --intrinsic_coeff 0.05 --max_step 1 --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_4_seed2.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_steps_before_training_25000 --intrinsic_coeff 0.05 --max_step 1 --min_num_steps_before_training 25000 > nohup_logs/surprise_ant_corridor_goal4_5_seed2.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_decay --intrinsic_coeff 0.05 --max_step 200000 --int_coeff_decay --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_6_seed1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_decay --intrinsic_coeff 0.05 --max_step 200000 --int_coeff_decay --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_6_seed2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --env_name reward_ant_corridor_resource_env_v0 --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_reward10 --intrinsic_coeff 0.05 --max_step 200000 --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_7_seed1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --env_name reward_ant_corridor_resource_env_v0 --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_reward10 --intrinsic_coeff 0.05 --max_step 200000 --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_7_seed2.txt 2>&1 &


# 30号的实验
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_normal --intrinsic_coeff 0.05 --max_step 200000 --min_num_steps_before_training 5000 --intrinsic_normal > nohup_logs/surprise_ant_corridor_goal4_6_intrinsic_normal_seed1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_normal --intrinsic_coeff 0.05 --max_step 200000 --min_num_steps_before_training 5000 --intrinsic_normal > nohup_logs/surprise_ant_corridor_goal4_6_intrinsic_normal_seed2.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_decay_100000 --intrinsic_coeff 0.05 --max_step 100000 --int_coeff_decay --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_5_maxstep_100000_seed1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_decay_100000 --intrinsic_coeff 0.05 --max_step 100000 --int_coeff_decay --min_num_steps_before_training 5000 > nohup_logs/surprise_ant_corridor_goal4_5_maxstep_100000_seed2.txt 2>&1 &



CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_normal_int01 --intrinsic_coeff 0.1 --max_step 200000 --min_num_steps_before_training 5000 --intrinsic_normal > nohup_logs/surprise_ant_corridor_goal4_6_intrinsic_01_normal_seed1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/ant_corridor/surprise/ant_corridor/surprise_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/surprise_normal_int01 --intrinsic_coeff 0.1 --max_step 200000 --min_num_steps_before_training 5000 --intrinsic_normal > nohup_logs/surprise_ant_corridor_goal4_6_intrinsic_01_normal_seed2.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python scripts/run.py configs/surprise-based/ant_corridor/surprise_vision/ant_corridor/surprise_vision_ant_corridor_goal4.json --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/ant_corridor/int_decay_test/goal4/test

!