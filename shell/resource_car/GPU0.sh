#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision_small_model_fuel_car_alpha15_fuel6.json > surprise_vision_small_model_fuel_car_alpha15_fuel61.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_action_cost_v1 --repeat 1 > surprise_ant_action_cost.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model_fuel_car.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 2 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/surprise_fuel_limited_fix_done_bug  > surprise_fuel_limited_fix_done_bug1fuel_mountain_car_done_fuel15_v0.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_corridor_fuel_done_30_goal9_v2 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_cheetah_corridor_fuel_done_30_goal9_v3  > surprise_cheetah_corridor_fuel_done_30_goal9_v3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_cheetah_corridor_fuel_done_32_goal9_v3 > surprise_cheetah_corridor_fuel_done_32_goal9_v3.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_fuel_done_150_v4 --repeat 1 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/surprise_vision_fuel_150 > surprise_vision_fuel_150.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --alg_type information_gain --ensemble_size 32 --env_name ant_corridor_fuel_done_140_v5 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/ig_fuel_140 > ig_fuel_1401.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --alg_type information_gain --ensemble_size 32 --env_name ant_corridor_fuel_done_140_v5 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/ig_fuel_140 > ig_fuel_1402.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --alg_type information_gain --ensemble_size 32 --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/ig_fuel_32 > ig_fuel_321.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --alg_type information_gain --ensemble_size 32 --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/ig_fuel_32 > ig_fuel_322.txt 2>&1 &
# sleep 15s
CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src_small_model.json --alg_type surprise --env_name fuel_mountain_car_done_fuel12_v1 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/src_fuel_12 > src_fuel_121.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src_small_model.json --alg_type surprise --env_name fuel_mountain_car_done_fuel12_v1 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/src_fuel_12 > src_fuel_122.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_resource_bonus_small_model.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/surprise_resource_bonus > surprise_resource_bonus1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_resource_bonus_small_model.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/surprise_resource_bonus > surprise_resource_bonus2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model_resource_costs.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/sac_resource_costs > sac_resource_costs1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model_resource_costs.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/sac_resource_costs > sac_resource_costs2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model_add_resource_bonus.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/sac_resource_bonus > sac_resource_bonus1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model_add_resource_bonus.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/sac_resource_bonus > sac_resource_bonus2.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_fuel_done_150_v4 --repeat 1 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/surprise_fuel_150 > surprise_fuel_150.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_fuel_done_140_v5 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/surprise_fuel_140 > surprise_fuel_1401.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_fuel_done_140_v5 --repeat 2 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/surprise_fuel_140 > surprise_fuel_1402.txt 2>&1 &




# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_fuel_done_160_v3 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/ant/surprise_vision_fuel_160  > surprise_vision_fuel_1601.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model_fuel_car.json --env_name fuel_mountain_car_done_fuel15_v0 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/surprise_fuel_limited_fix_done_bug  > surprise_fuel_limited_fix_done_bug2fuel_mountain_car_done_fuel12_v12.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model_fuel_car_alpha15_fuel6.json --env_name fuel_mountain_car_done_fuel15_v0 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/surprise_vision > surprise_vision_small_model_fuel_car_alpha15_fuel621.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model_fuel_car_alpha15_fuel6.json --env_name fuel_mountain_car_done_fuel15_v0 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/surprise_vision > surprise_vision_small_model_fuel_mountain_car_done_fuel12_v12.txt 2>&1 &


# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_small_model_fuel_car.json --env_name fuel_mountain_car_r100_v0 --repeat 3 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/car_action_cost_comparsion/sac_fuel_limited > sac_fuel_limited.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac_small_model_fuel_car.json --env_name continuous_mountaincar_action_cost --repeat 3 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/car_action_cost_comparsion/sac > sac.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_fuel_80_goal4_v2  --repeat 4 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/envs_fuel/surprise > surprise_ant_corridor_fuel_80_goal4_v2.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_action_cost_v2   --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_TO_IJCAI_data/envs_action_cost/surprise > ant_corridor_action_cost_v2.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_small_model_car_action_cost.json > surprise_small_model_car_action_cost.txt 2>&1 &