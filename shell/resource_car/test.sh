for env_name in "ant_corridor_fuel_done_140_v5" "cheetah_corridor_fuel_done_32_goal9_v3" "fuel_mountain_car_done_fuel12_v1"
do 
    if [ $env_name == "fuel_mountain_car_done_fuel12_v1" ]
    then
        echo "car"
        CUDA_VISIBLE_DEVICES=4 xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac_small_model.json --env_name fuel_mountain_car_done_fuel12_v1 --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/car/sac_fuel_12
    else 
        echo $env_name
        CUDA_VISIBLE_DEVICES=4 xvfb-run -a -s "-screen 0 1400x900x24" python scripts/run.py configs/sac/sac.json --env_name $env_name --repeat 3 --base_log_dir /home/zhwang/research/ICML_TO_IJCAI_data/data/envs_fuel_fix_r_done_bug/cheetah_or_ant/sac_fuel_12 
    fi
done