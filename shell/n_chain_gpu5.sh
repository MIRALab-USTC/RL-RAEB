#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python scripts/run.py configs/surprise-based/surprise_nchain.json
sleep 10s