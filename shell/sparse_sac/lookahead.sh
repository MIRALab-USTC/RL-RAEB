#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_cheetah.json > o_cheetah_lookahead1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_cheetah.json > o_cheetah_lookahead2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_cheetah.json > o_cheetah_lookahead3.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=2 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_hopper.json > o_hopper_lookahead1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_hopper.json > o_hopper_lookahead2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=3 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_hopper.json > o_hopper_lookahead3.txt 2>&1 &

sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_human.json > o_human_lookahead1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_human.json > o_human_lookahead2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=5 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_human.json > o_human_lookahead3.txt 2>&1 &

sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_walker.json > o_walker_lookahead1.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=6 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_walker.json > o_walker_lookahead2.txt 2>&1 &
sleep 10s
CUDA_VISIBLE_DEVICES=7 nohup python scripts/run.py configs/surprise-based-max-state_entropy/sac_lookahead_novelty/sac_lookahead_walker.json > o_walker_lookahead3.txt 2>&1 &