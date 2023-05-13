# Efficient Exploration in Resource-Restricted Reinforcement Learning

This is the code of paper 
**Efficient Exploration in Resource-Restricted Reinforcement Learning
**. 
Zhihai Wang*, Taoxing Pan*, Qi Zhou, Jie Wang**. AAAI 2023. [[arXiv](https://arxiv.org/abs/2212.06988)]

## Requirements
- Python 3.6.9
- PyTorch 1.10
- tqdm
- gym 0.21
- mujoco 1.50
```
pip install -r requirements.txt
```

## Reproduce the Results
1. For example, run experiments on Ant 
```
python scripts/run.py configs/surprise_based/surprise_vision.json
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@misc{wang2022efficient,
      title={Efficient Exploration in Resource-Restricted Reinforcement Learning}, 
      author={Zhihai Wang and Taoxing Pan and Qi Zhou and Jie Wang},
      year={2022},
      eprint={2212.06988},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Other Repositories
If you are interested in our work, you may find the following papers useful.

**Model-Based Reinforcement Learning via Estimated Uncertainty and Conservative Policy Optimization.**
*Qi zhou, Houqiang Li, Jie Wang*.* AAAI 2020. [[paper](https://arxiv.org/abs/1911.12574)] [[code](https://github.com/MIRALab-USTC/RL-POMBU)]

**Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic.**
*Zhihai Wang, Jie Wang*, Qi Zhou, Bin Li, Houqiang Li.* AAAI 2022. [[paper](https://arxiv.org/abs/2112.10504)] [[code](https://github.com/MIRALab-USTC/RL-CMBAC)]
