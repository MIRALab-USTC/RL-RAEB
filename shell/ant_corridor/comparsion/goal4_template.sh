#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name done_resource_cheetah_corridor_v3 --repeat 2 --num_eval_steps_per_epoch 4000 --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v3/sac > sac1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name done_resource_cheetah_corridor_v3 --repeat 3 --num_eval_steps_per_epoch 4000 --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v3/sac > sac2.txt 2>&1 &
# sleep 15s

# # CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name done_resource_cheetah_corridor_v4 --repeat 2 --num_eval_steps_per_epoch 4000 --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v4/sac > sac3.txt 2>&1 &
# # sleep 15s
# # CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name done_resource_cheetah_corridor_v4 --repeat 3 --num_eval_steps_per_epoch 4000 --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v4/sac > sac4.txt 2>&1 &
# # sleep 15s

# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name done_resource_cheetah_corridor_v3 --repeat 2 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v3/surprise > surprise1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name done_resource_cheetah_corridor_v3 --repeat 3 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v3/surprise > surprise2.txt 2>&1 &
# sleep 15s
# # CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name done_resource_cheetah_corridor_v4 --repeat 2 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v4/surprise > surprise3.txt 2>&1 &
# # sleep 15s
# # CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name done_resource_cheetah_corridor_v4 --repeat 3 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v4/surprise > surprise4.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name ant_fuel_cargo_v1 --repeat 5 > rnd_noveld_electric_delivery_ant.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name cheetah_fuel_cargo_v1 --repeat 5 > rnd_noveld_electric_delivery_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/rnd/rnd_small_noveld.json --env_name fuel_cargo_car_v1 --repeat 5 > rnd_noveld_electric_delivery_car.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/src.json --env_name ant_fuel_cargo_v1 --repeat 5 > lagrigian_electric_delivery_ant.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/src.json --env_name cheetah_fuel_cargo_v1 --repeat 5 > lagrigian_electric_delivery_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/src_small_model.json --env_name fuel_cargo_car_v1 --repeat 5 > lagrigian_electric_delivery_car.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/src.json --env_name ant_corridor_fuel_done_140_v5 --repeat 5 > lagrigian_electric_ant.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based/src.json --env_name cheetah_corridor_fuel_done_32_goal9_v3 --repeat 5 > lagrigian_electric_cheetah.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/src.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 5 > lagrigian_delivery_ant.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah5.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 1 > lagrigian_delivery_cheetah6.txt 2>&1 &

# 晚上跑delivery cheetah 
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat5.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name fuel_cargo_car_v1 --repeat 1 > raeb_edcat6.txt 2>&1 &




# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name resource_mountaincar_v0 --repeat 3 > rnd_noveld_delivery_car2.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision5.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0_v2 --repeat 1 --alg_type surprise --int_coeff 0.5 --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/evaluation/ant_corridor_resource_env_goal_4_v0_v2/surprise_vision > surprise_vision5.txt 2>&1 &



# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain > ant_corridor_env_42.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise5.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 1 --num_eval_steps_per_epoch 8000 --alg_type only_resource_bonus  --base_log_dir /home/zhwang/r3l_exploration/data/ablation/only_resource_bonus > surprise_vision_surprise6.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac/sac_add_resource_bonus.json --env_name ant_corridor_fuel_done_140_v5 --repeat 3 --num_eval_steps_per_epoch 8000 > sac_rb1.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/sac/sac_add_resource_bonus.json --env_name ant_corridor_fuel_done_140_v5 --repeat 3 --num_eval_steps_per_epoch 8000 > sac_rb2.txt 2>&1 &


## delivery ant 2 cargo
# raeb
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2_2cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_2_cargo > raeb_2_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2_8cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_8_cargo > raeb_8_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0_v2_16cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_16_cargo > raeb_16_cargo1.txt 2>&1 &
sleep 15s

# sac
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/sac/sac.json --env_name ant_corridor_resource_env_goal_4_v0_v2_2cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_2_cargo > sac_2_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/sac/sac.json --env_name ant_corridor_resource_env_goal_4_v0_v2_8cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_8_cargo > sac_8_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/sac/sac.json --env_name ant_corridor_resource_env_goal_4_v0_v2_16cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_16_cargo > sac_16_cargo1.txt 2>&1 &
sleep 15s


# surprise
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0_v2_2cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_2_cargo > surprise_2_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0_v2_8cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_8_cargo > surprise_8_cargo1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0_v2_16cargo --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_16_cargo > surprise_16_cargo1.txt 2>&1 &
sleep 15s

# noveld
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name ant_corridor_resource_env_goal_4_v0_v2_2cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_2_cargo > rnd_noveld_2cargo.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name ant_corridor_resource_env_goal_4_v0_v2_8cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_8_cargo > rnd_noveld_8cargo.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/rnd/rnd_noveld.json --env_name ant_corridor_resource_env_goal_4_v0_v2_16cargo --repeat 3 --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/r3l_exploration/aaai23_rebuttal_data/delivery_ant_16_cargo > rnd_noveld_16cargo.txt 2>&1 &
sleep 15s


# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name done_resource_cheetah_corridor_v4 --repeat 2 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v2/surprise_vision > surprise_vision_surprise3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name done_resource_cheetah_corridor_v4 --repeat 3 --num_eval_steps_per_epoch 4000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/abalation/with_done/done_resource_cheetah_corridor_v2/surprise_vision > surprise_vision_surprise4.txt 2>&1 &



# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name done_resource_mountaincar_v0 --repeat 2 --alg_type surprise  --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/abalation/with_done/done_resource_mountaincar_v0/surprise > surprise5.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name done_resource_mountaincar_v0 --repeat 3 --alg_type surprise  --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/abalation/with_done/done_resource_mountaincar_v0/surprise > surprise6.txt 2>&1 &
# sleep 15s


# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json --env_name done_resource_mountaincar_v0 --repeat 2  --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/abalation/with_done/done_resource_mountaincar_v0/sac > sac6.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json --env_name done_resource_mountaincar_v0 --repeat 3  --base_log_dir /home/rl_shared/zhihaiwang/research/ICML_data/abalation/with_done/done_resource_mountaincar_v0/sac > sac7.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.02 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v01.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v02.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.15 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v03.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.002 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v04.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.02 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v05.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v06.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.15 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v07.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.002 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v08.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.02 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v09.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v010.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.15 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v011.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 1 --alg_type surprise --int_coeff 0.002 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v012.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --alg_type surprise --int_coeff 0.02 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v013.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --alg_type surprise --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v014.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --alg_type surprise --int_coeff 0.15 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v015.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --alg_type surprise --int_coeff 0.002 --base_log_dir /home/zhwang/research/ICML_data/ablation/resource_cheetah_corridor_v0/surprise_vision > resource_cheetah_corridor_v016.txt 2>&1 &




# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_long_steps/resource_cheetah_corridor_v0/simhash_vision > simhash_vision_ant_corridor1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name resource_cheetah_corridor_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_long_steps/resource_cheetah_corridor_v0/simhash_vision > simhash_vision_ant_corridor2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash.json --env_name resource_cheetah_corridor_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_long_steps/resource_cheetah_corridor_v0/simhash > simhash_ant_corridor1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash.json --env_name resource_cheetah_corridor_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_long_steps/resource_cheetah_corridor_v0/simhash > simhash_ant_corridor2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain > ant_corridor_env_42.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain > ant_corridor_env_41.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain_vision > ant_corridor_env_43.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain_vision > ant_corridor_env_44.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/valid_src/goal4/src > src1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/valid_src/goal4/src > src2.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/valid_src/resource_cheetah_corridor_v0/src > src3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=2 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src.json --env_name resource_cheetah_corridor_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/valid_src/resource_cheetah_corridor_v0/src > src4.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=5 xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/src.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --alg_type surprise  --base_log_dir /home/zhwang/research/ICML_data/valid_src/goal4/src


# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/rnd/rnd.json --env_name ant_corridor_resource_env_goal_6_v0 --repeat 5 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/continuous_resource_25/information_gain > continuous_resource_251.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/rnd/rnd.json --env_name ant_corridor_resource_env_goal_6_v0 --repeat 5   --base_log_dir /home/zhwang/research/ICML_data/valid_rnd/goal6/rnd > goal6_rnd1.txt 2>&1 &
# sleep 15s

# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/rnd/rnd.json --env_name ant_corridor_resource_env_goal_5_v0 --repeat 2  --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/research/ICML_data/valid_rnd/goal5/rnd > goal5_rnd2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/rnd/rnd.json --env_name ant_corridor_resource_env_goal_5_v0 --repeat 3  --num_eval_steps_per_epoch 8000 --base_log_dir /home/zhwang/research/ICML_data/valid_rnd/goal5/rnd > goal5_rnd3.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 2 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain_vision > ant_corridor_env_47.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v0 --repeat 3 --alg_type information_gain --ensemble_size 32  --base_log_dir /home/zhwang/research/ICML_data/valid_ig/ant_corridor_resource_env_goal_4_v0/information_gain_vision > ant_corridor_env_48.txt 2>&1 &




# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name mountain_car_v1 --repeat 2 --train_model_freq 999 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/research/ICML_data/test_pe/mountaincar/information_gain_small_model > pe_low_freq1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name mountain_car_v1 --repeat 2 --train_model_freq 999 --alg_type information_gain --ensemble_size 32 --base_log_dir /home/zhwang/research/ICML_data/test_pe/mountaincar/information_gain_small_model > pe_low_freq2.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.25 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_ant_corridor_goal_4/simhash_vision > simhash_vision_ant_corridor3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.25 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_ant_corridor_goal_4/simhash_vision > simhash_vision_ant_corridor4.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 18 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision_k18 > simhash_vision_k18_15541.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 18 --int_coeff 0.05 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision_k18 > simhash_vision_k182.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 18 --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision_k18 > simhash_vision_k183.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=6 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 18 --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision_k18 > simhash_vision_k184.txt 2>&1 &
# # sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision > simhash_vision_15545.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 2 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.1 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash_adjust_intcoeff/resource_ant_corridor_goal_4/simhash_vision > simhash_vision_15546.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=5 xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name ant_corridor_resource_env_goal_4_v0 --repeat 3 --num_eval_steps_per_epoch 8000 --max_path_length 500 --hash_k 20 --int_coeff 0.01 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_ant_corridor_goal_4_test/simhash_vision


# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_env_4 --repeat 2 --alg_type prediction_error --base_log_dir /home/zhwang/research/ICML_data/test_pe/ant_corridor_goal4/pe > pe1.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name ant_corridor_env_4 --repeat 2 --alg_type prediction_error --base_log_dir /home/zhwang/research/ICML_data/test_pe/ant_corridor_goal4/pe > pe2.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name mountain_car_v1 --repeat 2 --alg_type prediction_error --base_log_dir /home/zhwang/research/ICML_data/test_pe/mountaincar/pe > pe3.txt 2>&1 &
# sleep 15s
# CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_small_model.json --env_name mountain_car_v1 --repeat 2 --alg_type prediction_error --base_log_dir /home/zhwang/research/ICML_data/test_pe/mountaincar/pe > pe4.txt 2>&1 &

:<<!
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash.json --env_name resource_mountaincar_v0 --repeat 2 --num_eval_steps_per_epoch 4000 --max_path_length 200 --hash_k 16 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_mountaincar_10/simhash > simhash1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash.json --env_name resource_mountaincar_v0 --repeat 3 --num_eval_steps_per_epoch 4000 --max_path_length 200 --hash_k 16 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_mountaincar_10/simhash > simhash2.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name resource_mountaincar_v0 --repeat 2 --num_eval_steps_per_epoch 4000 --max_path_length 200 --hash_k 16 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_mountaincar_10/simhash_vision > simhash_vision1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac_hash/simhash_vision.json --env_name resource_mountaincar_v0 --repeat 3 --num_eval_steps_per_epoch 4000 --max_path_length 200 --hash_k 16 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/resource_mountaincar_10/simhash_vision > simhash_vision2.txt 2>&1 &
sleep 15s


CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name ant_corridor_env_4 --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/valid_simhash/ant_corridor_goal4/sac > sac3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name no_reward_resource_ant_corridor --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/analysis_with_pool/ant_corridor_no_reward/surprise > surprise_pool.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=7 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name no_reward_resource_ant_corridor --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/analysis_with_pool/ant_corridor_no_reward/surprise_vision > surprise_vision_pool.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name resource_cheetah_corridor_v0 --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real_addx/goal4/sac > sac1.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name resource_cheetah_corridor_v0 --repeat 3 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real_addx/goal4/sac > sac2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup  xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v0 --repeat 2 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real_addx/goal4/surprise > surprise1.txt 2>&1 &
sleep 15s 
CUDA_VISIBLE_DEVICES=1 nohup  xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real_addx/goal4/surprise > surprise2.txt 2>&1 &
sleep 15s 
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real/goal4/sac > sac1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real/goal4/sac > sac2.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup  xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v1 --intrinsic_normal --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real/goal3/surprise_int_normal > surprise_int_normal1.txt 2>&1 &
#sleep 15s
#CUDA_VISIBLE_DEVICES=1 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise_vision.json --env_name resource_cheetah_corridor_v1 --intrinsic_normal --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor_real/goal3/surprise_vision_int_normal > surprise_vision_int_normal1.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=3 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/surprise-based/surprise.json --env_name resource_cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/resource_cheetah_corridor/goal4/surprise > surprise2.txt 2>&1 &

#CUDA_VISIBLE_DEVICES=0 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name cheetah_corridor_v0 --base_log_dir /home/zhwang/research/ICML_data/cheetah/corridor/goal4/sac > sac1.txt 2>&1 &


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