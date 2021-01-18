source activate rl_tx
cd /home/txpan/mbrl_exploration_with_novelty
python scripts/run.py configs/sac/sac_ant_maze/ant_corridor/sac_goal6.json

CUDA_VISIBLE_DEVICES=6 nohup python -u scripts/run.py configs/rnd/rnd.json --repeat 3 --env_name resource_cheetah_corridor_v0 --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/resource_cheetah_corridor/goal4/rnd >nohup_log/cheetah_corridor/rnd_210116_1453_repeat3 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python -u scripts/run.py configs/rnd/rnd.json --repeat 2 --env_name resource_cheetah_corridor_v0 --base_log_dir /home/txpan/research/ICML_data/exploration_env_exps_fix_env/resource_cheetah_corridor/goal4/rnd >nohup_log/cheetah_corridor/rnd_210116_1455_repeat2 2>&1 &