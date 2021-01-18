
#!/bin/bash
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json  --repeat 5 --env_name resource_mountaincar_v4 --base_log_dir /home/zhwang/research/ICML_data/continuous_resource_5/sac > sac_resource5.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=4 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json  --repeat 5 --env_name resource_mountaincar_v0 --base_log_dir /home/zhwang/research/ICML_data/continuous_resource_10/sac > sac_resource10.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json  --repeat 5 --env_name resource_mountaincar_v8 --base_log_dir /home/zhwang/research/ICML_data/continuous_resource_15/sac > sac_resource15.txt 2>&1 &
sleep 15s
CUDA_VISIBLE_DEVICES=5 nohup xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json  --repeat 5 --env_name resource_mountaincar_v3 --base_log_dir /home/zhwang/research/ICML_data/continuous_resource_25/sac > sac_resource25.txt 2>&1 &
