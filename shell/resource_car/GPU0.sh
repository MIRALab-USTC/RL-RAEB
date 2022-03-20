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
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name cheetah_corridor_fuel_done_30_goal9_v2 --repeat 3 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_vision_fuel_30 > surprise_vision_fuel_30.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_vision_fuel_32 > surprise_vision_fuel_321.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_vision_fuel_32 > surprise_vision_fuel_322.txt 2>&1 &
# sleep 15s
# # CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_corridor_fuel_done_30_goal9_v2 --repeat 3 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_fuel_30 > surprise_fuel_30.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah/surprise_fuel_32 > surprise_fuel_32.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json --env_name fuel_mountain_car_done_fuel12_v1  --repeat 2 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/only_resource_bonus  > only_resource_bonus1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision_small_model.json --env_name fuel_mountain_car_done_fuel12_v1  --repeat 2 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/only_resource_bonus  > only_resource_bonus2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo/vision --env_name cheetah_fuel_cargo_v1 --repeat 2 > raeb1_fuel_cargo_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo/vision --env_name cheetah_fuel_cargo_v1 --repeat 2 > raeb2_fuel_cargo_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/surprise.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo/surprise --env_name cheetah_fuel_cargo_v1 --repeat 2 > surprise1_fuel_cargo_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based/surprise.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo/surprise --env_name cheetah_fuel_cargo_v1 --repeat 2 > surprise2_fuel_cargo_cheetah.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/ant_goal4_cargo/vision_int005 --env_name ant_goal4_cargo_resource_beta1 --repeat 2 --intrinsic_coeff 0.05 > ant_goal4_cargo_resource_beta11.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/ant_goal4_cargo/vision_int005 --env_name ant_goal4_cargo_resource_beta1 --repeat 2 --intrinsic_coeff 0.05 > ant_goal4_cargo_resource_beta12.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc1.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc1 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v01.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc1.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc1 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v02.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc001.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc001 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v03.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc001.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc001 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v04.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc05.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc05 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v05.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_resource_bonus_rbc05.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cargo_ant_surpriserb/rbc05 --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 > ant_corridor_resource_env_goal_4_v06.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/src.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/IJCAI22_rebuttal/more_comparsion_methods/electric_cheetah/lagrangian > src1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/src.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/IJCAI22_rebuttal/more_comparsion_methods/electric_cheetah/lagrangian > src2.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/simhash --repeat 1 > ant_fuel_cargo_v1hash1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/simhash --repeat 1 > ant_fuel_cargo_v1hash2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/simhash --repeat 1 > ant_fuel_cargo_v1hash3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/simhash --repeat 1 > ant_fuel_cargo_v1hash4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name cheetah_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/simhash --repeat 1 > cheetah_fuel_cargo_v11hash.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name cheetah_fuel_cargo_v1 --base_log_dir  /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/simhash --repeat 1 > cheetah_fuel_cargo_v12hash.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/sac --env_name ant_fuel_cargo_v1 --repeat 2 > ant_fuel_cargo_v11.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup python scripts/run.py configs/sac/sac.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/ant_fuel_cargo_v1/sac --env_name ant_fuel_cargo_v1 --repeat 2 > ant_fuel_cargo_v12.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/sac --env_name cheetah_fuel_cargo_v1 --repeat 2 > cheetah_fuel_cargo_v11.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac/sac.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/sac --env_name cheetah_fuel_cargo_v1 --repeat 2 > cheetah_fuel_cargo_v12.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_small_model.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/sac --env_name fuel_cargo_car_v1 --repeat 2 > fuel_cargo_car_v11.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac/sac_small_model.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/sac --env_name fuel_cargo_car_v1 --repeat 2 > fuel_cargo_car_v12.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_fuel_cargo_v1 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/information_gain > cheetah_fuel_cargo_v1_information_gain1.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name cheetah_fuel_cargo_v1 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/cheetah_fuel_cargo_v1/information_gain > cheetah_fuel_cargo_v1_information_gain2.txt 2>&1 &
# sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name fuel_cargo_car_v1 --repeat 1 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/information_gain > fuel_cargo_car_v1_information_gain1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name fuel_cargo_car_v1 --repeat 1 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/information_gain > fuel_cargo_car_v1_information_gain2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name fuel_cargo_car_v1 --repeat 1 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/information_gain > fuel_cargo_car_v1_information_gain3.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name fuel_cargo_car_v1 --repeat 1 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/fuel_cargo_car_v1/information_gain > fuel_cargo_car_v1_information_gain4.txt 2>&1 &



# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_small_model_fuel_car.json > surprise2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_small_model_fuel_car.json > surprise3.txt 2>&1 &




# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/cheetah_cargo_goal9/simhash --repeat 2 > cheetah_cargo_goal91.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/cheetah_cargo_goal9/simhash --repeat 2 > cheetah_cargo_goal92.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_corridor_resource_env_goal_4_v0 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/ant_cargo_goal4/simhash --repeat 2 > ant_cargo_goal41.txt 2>&1 &
# sleep 15s
# # CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_corridor_resource_env_goal_4_v0 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/ant_cargo_goal4/simhash --repeat 2 > ant_cargo_goal42.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_corridor_fuel_done_140_v5 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/ant_fuel/simhash --repeat 2 > ant_fuel1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name ant_corridor_fuel_done_140_v5 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/ant_fuel/simhash --repeat 2 > ant_fuel2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/cheetah_fuel/simhash --repeat 2 > cheetah_fuel1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac_hash/simhash.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --base_log_dir /home/zhwang/IJCAI2022_rebuttal/data/cheetah_fuel/simhash --repeat 2 > cheetah_fuel2.txt 2>&1 &



# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/sac_hash/simhash_small.json --env_name resource_mountaincar_v0 > simhash2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac_hash/simhash_small.json --env_name resource_mountaincar_v0 > simhash3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac_hash/simhash_small.json --env_name resource_mountaincar_v0 > simhash4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac_hash/simhash_small.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/delivery_ant/simhash --env_name ant_corridor_resource_env_goal_4_v0 > simhash_deli_ant1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/sac_hash/simhash_small.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/delivery_ant/simhash --env_name ant_corridor_resource_env_goal_4_v0 > simhash_deli_ant2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac_hash/simhash_small.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/delivery_ant/simhash --env_name ant_corridor_resource_env_goal_4_v0 > simhash_deli_ant3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/sac_hash/simhash_small.json --base_log_dir /home/zhwang/ICML_TO_IJCAI/data/delivery_ant/simhash --env_name ant_corridor_resource_env_goal_4_v0 > simhash_deli_ant4.txt 2>&1 &



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